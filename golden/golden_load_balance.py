import numpy as np
import os, glob, shutil, subprocess
import math
from dataclasses import dataclass

# ---------------- Common helpers ----------------

def tile_matrix(matrix, row_tiles, col_tiles):
    rows, cols = matrix.shape
    assert rows % row_tiles == 0 and cols % col_tiles == 0
    reshaped = matrix.reshape(rows // row_tiles, row_tiles,
                              cols // col_tiles, col_tiles)
    transposed = reshaped.transpose(0, 2, 1, 3)
    return transposed.flatten()

def _save_vec16(path, vec_1d: np.ndarray, iterations: int):
    vec = vec_1d.reshape(-1)
    pad = (-len(vec)) % 16
    if pad:
        vec = np.pad(vec, (0, pad), mode="constant")
    tiled = np.tile(vec, (iterations, 1)).reshape(-1, 16)
    np.savetxt(path, tiled, fmt="%s", delimiter=" ")


# ---------------- Graph utilities ----------------

class EmitContext:
    def __init__(self, graph_path):
        self.graph_path = graph_path
        self._conn_idx = 0
        self._sb_idx = 0
        self.kernel_sigs = []

    def _write(self, text):
        with open(self.graph_path, "a") as f:
            f.write(text + "\n")

    def new_conn(self):
        name = f"c{self._conn_idx}"
        self._conn_idx += 1
        return name

    def register_kernel(self, sig: "KernelSignature"):
        self.kernel_sigs.append(sig)

    def connect_stream(self, src_expr, dst_expr, fifo=32, allow_single_buffer=False):
        conn = self.new_conn()
        self._write(f"auto {conn} = connect<stream>({src_expr}, {dst_expr});")
        if fifo:
            self._write(f"fifo_depth({conn}) = {fifo};")
        if allow_single_buffer:
            self._write(f"single_buffer({dst_expr});")
        return conn

    def create_shared_buffer(self, num_bytes, num_inputs, num_outputs, prefix):
        name = f"{prefix}{self._sb_idx}"
        self._sb_idx += 1
        self._write(
            f"auto {name} = adf::shared_buffer<int8>::create({{{num_bytes}}}, {num_inputs}, {num_outputs});"
        )
        return name

    def set_access(self, buf_name, direction, port_idx, length, offset):
        tp = f"{buf_name}_tp_{direction}_{port_idx}"
        ap = f"{buf_name}_ap_{direction}_{port_idx}"
        self._write(f"adf::tiling_parameters {tp};")
        self._write(f"{tp}.buffer_dimension = {{{length}}};")
        self._write(f"{tp}.tiling_dimension = {{{length}}};")
        self._write(f"{tp}.offset = {{{offset}}};")
        self._write(
            f"{tp}.tile_traversal.push_back(adf::traversing_parameters{{0, {length}, 1}});"
        )
        self._write(f"adf::access_pattern {ap}({tp});")
        if direction == "write":
            self._write(f"adf::write_access({buf_name}.in[{port_idx}]) = {{ {ap} }};")
        else:
            self._write(f"adf::read_access({buf_name}.out[{port_idx}]) = {{ {ap} }};")


@dataclass
class KernelSignature:
    idx: int
    inputs: list  # e.g., ["stream_int8"]
    outputs: list


# ---------------- Layers ----------------

class Layer:
    def emit(self, layer_idx, kernel_idx, src_expr, src_kind, src_bytes,
             x_in, y_ref, iterations, ctx):
        raise NotImplementedError
    def forward(self, x_in):
        raise NotImplementedError


class Dense(Layer):
    def __init__(self, N, shift=0, relu=False, m_tile=2, k_tile=8, n_tile=8, o_tiles=1):
        self.N = N
        self.shift = shift
        self.relu = relu
        self.m_tile = m_tile
        self.k_tile = k_tile
        self.n_tile = n_tile
        self.o_tiles = o_tiles
        self._last_in2d = None

    def _as_2d(self, x_in):
        if x_in.ndim == 2:
            return x_in
        if x_in.ndim == 3:
            XH, XW, XC = x_in.shape
            return x_in.reshape(XH * XW, XC)
        raise ValueError(f"Dense expects 2D or 3D input, got {x_in.shape}")

    def forward(self, x_in):
        x2d = self._as_2d(x_in)
        self._last_in2d = x2d
        R, K = x2d.shape
        assert (R % self.m_tile) == 0 and (K % self.k_tile) == 0
        assert self.N % (self.n_tile * self.o_tiles) == 0
        self.K = K
        self.N_per_tile = self.N // self.o_tiles
        self.W = np.random.randint(0, 128, size=(K, self.N), dtype=np.int8)
        y = (x2d.astype(np.int32) @ self.W.astype(np.int32))
        y = (y >> self.shift).astype(np.int8)
        if self.relu:
            y = np.maximum(0, y)
        return y

    def emit(self, layer_idx, kernel_idx, src_expr, src_kind, src_bytes,
             x_in, y_ref, iterations, ctx):
        assert src_kind == "stream"
        m, k, n = self.m_tile, self.k_tile, self.n_tile
        x2d = self._last_in2d if self._last_in2d is not None else self._as_2d(x_in)

        x_tiled = tile_matrix(x2d, m, k)
        a_tiled = tile_matrix(y_ref, m, n)
        np.savetxt(f"data/x{layer_idx}.txt",
                   np.tile(x_tiled, (iterations, 1)).reshape(-1, 16),
                   fmt="%s", delimiter=" ")
        np.savetxt(f"data/a{layer_idx}.txt",
                   np.tile(a_tiled, (iterations, 1)).reshape(-1, 16),
                   fmt="%s", delimiter=" ")

        def emit_dense_kernel(idx, W_slice, mm_N):
            k_tiled = tile_matrix(W_slice, k, n)
            with open(f"model/layer_{idx}.cc", "a") as f:
                f.write(f'''
#define mm_m {m}
#define mm_k {k}
#define mm_n {n}
#define mm_M {x2d.shape[0]}
#define mm_K {x2d.shape[1]}
#define mm_N {mm_N}
#define SHIFT {self.shift}
#define DO_RELU {str(self.relu).lower()}

#include <cstdint>

__attribute__((section(".data"))) alignas(32) int8_t matB [{k_tiled.size}] = {{ {", ".join(str(int(x)) for x in k_tiled)} }};

#include "dense_i8_stream.h"

void f{idx}(input_stream_int8 * __restrict in, output_stream_int8 * __restrict out){{ dense_i8(in, out);}}
''')
            ctx._write(f"layers[{idx}] = kernel::create(f{idx});")
            ctx._write(f'source(layers[{idx}]) = "layer_{idx}.cc";')
            ctx.register_kernel(KernelSignature(idx,
                                                inputs=["stream_int8"],
                                                outputs=["stream_int8"]))

        total_in_bytes = src_bytes
        allow_single = (src_expr == "AIE_IN.out[0]" and total_in_bytes > 32768)

        if self.o_tiles == 1:
            emit_dense_kernel(kernel_idx, self.W, self.N)
            ctx.connect_stream(src_expr, f"layers[{kernel_idx}].in[0]",
                               allow_single_buffer=allow_single)
            out_expr = f"layers[{kernel_idx}].out[0]"
            out_bytes = y_ref.size * y_ref.itemsize
            return kernel_idx + 1, out_expr, "stream", out_bytes

        # shared buffer to broadcast input stream
        sb_in = ctx.create_shared_buffer(total_in_bytes, 1, self.o_tiles,
                                         prefix=f"sb_in_{layer_idx}_")
        ctx.set_access(sb_in, "write", 0, total_in_bytes, 0)
        for t in range(self.o_tiles):
            ctx.set_access(sb_in, "read", t, total_in_bytes, 0)
        ctx.connect_stream(src_expr, f"{sb_in}.in[0]", allow_single_buffer=allow_single)

        bytes_per_branch = x2d.shape[0] * self.N_per_tile * np.dtype(np.int8).itemsize
        branch_exprs = []
        for tile in range(self.o_tiles):
            dense_idx = kernel_idx
            kernel_idx += 1
            n_begin = tile * self.N_per_tile
            n_end = (tile + 1) * self.N_per_tile
            emit_dense_kernel(dense_idx, self.W[:, n_begin:n_end], self.N_per_tile)
            ctx.connect_stream(f"{sb_in}.out[{tile}]",
                               f"layers[{dense_idx}].in[0]")
            branch_exprs.append(f"layers[{dense_idx}].out[0]")

        total_out_bytes = y_ref.size * y_ref.itemsize
        sb_out = ctx.create_shared_buffer(total_out_bytes, self.o_tiles, 1,
                                          prefix=f"sb_out_{layer_idx}_")
        for t in range(self.o_tiles):
            ctx.set_access(sb_out, "write", t, bytes_per_branch, t*bytes_per_branch)
        ctx.set_access(sb_out, "read", 0, total_out_bytes, 0)

        for t, expr in enumerate(branch_exprs):
            ctx.connect_stream(expr, f"{sb_out}.in[{t}]")

        out_expr = f"{sb_out}.out[0]"
        out_bytes = total_out_bytes
        return kernel_idx, out_expr, "stream", out_bytes


# ---------------- Sequential-ish model ----------------

class Sequential:
    def __init__(self, iterations=1):
        self.layers = []
        self.iterations = iterations

    def add(self, layer: Layer):
        self.layers.append(layer)

    def build_and_emit(self, x0: np.ndarray):
        graph_path = "model/layer_graph.h"
        open(graph_path, "w").close()
        ctx = EmitContext(graph_path)

        x = x0
        kernel_idx = 0
        src_expr = "AIE_IN.out[0]"
        src_kind = "stream"
        src_bytes = x0.size * x0.itemsize

        for layer_idx, layer in enumerate(self.layers):
            y = layer.forward(x)
            kernel_idx, src_expr, src_kind, src_bytes = layer.emit(
                layer_idx, kernel_idx, src_expr, src_kind, src_bytes,
                x, y, self.iterations, ctx
            )
            x = y

        N_LAYERS = kernel_idx
        in_bytes = x0.size * x0.itemsize

        last = self.layers[-1]
        if isinstance(last, Dense):
            tiled_last = tile_matrix(x, last.m_tile, last.n_tile)
            np.savetxt("data/out_ref.txt",
                       np.tile(tiled_last, (self.iterations,1)).reshape(-1,16),
                       fmt="%s", delimiter=" ")
        else:
            _save_vec16("data/out_ref.txt", x.flatten(), self.iterations)

        ctx.connect_stream(src_expr, "AIE_OUT.in[0]")

        with open("model/include.h", "w") as f:
            f.write(f"#define N_LAYERS {N_LAYERS}\n")
            f.write(f"#define ITERATIONS {self.iterations}\n")
            out_bytes = x.size * x.itemsize
            f.write(f"#define TOT_OUT_BYTES {out_bytes*self.iterations}\n")
            f.write(f"#define TOT_IN_BYTES {in_bytes*self.iterations}\n")

            for sig in sorted(ctx.kernel_sigs, key=lambda s: s.idx):
                params = []
                for spec in sig.inputs:
                    params.append("input_stream_int8 * __restrict")
                for spec in sig.outputs:
                    params.append("output_stream_int8 * __restrict")
                f.write(f"void f{sig.idx}({', '.join(params)});\n")

        return x


# ---------------- Main ----------------

if __name__ == "__main__":
    m_tile = 2
    k_tile = 8
    n_tile = 8
    iterations = 1

    for path in [
        "data", "model",
        "*.log", "aiesimulator_output", "Work", ".Xil",
        ".AIE_SIM_CMD_LINE_OPTIONS", "ISS_RPC_SERVER_PORT",
        "libadf.a", "Map_Report.csv", "pl_sample_counts",
        "plio_throughput_info.json", "sol.db", "aiesim.vcd"
    ]:
        for p in glob.glob(path):
            if os.path.isdir(p):
                shutil.rmtree(p, ignore_errors=True)
            elif os.path.exists(p):
                os.remove(p)
    os.makedirs("data", exist_ok=True)
    os.makedirs("model", exist_ok=True)

    BATCH = 4
    INPUTS = 128

    x0 = np.random.randint(0, 128, size=(BATCH, INPUTS), dtype=np.int8)
    model = Sequential(iterations=iterations)
    model.add(Dense(N=128, shift=5, relu=False,
                    m_tile=m_tile, k_tile=k_tile, n_tile=n_tile, o_tiles=4))
    model.add(Dense(N=16, shift=2, relu=False,
                    m_tile=m_tile, k_tile=k_tile, n_tile=n_tile, o_tiles=1))

    y_ref_final = model.build_and_emit(x0)

    subprocess.run(["./run.sh"], check=True)

    aie_out_path = "aiesimulator_output/data/out_sim.txt"
    assert os.path.exists(aie_out_path), f"Missing {aie_out_path}"
    with open(aie_out_path, "r") as infile, open("data/out_sim.txt", "w") as outfile:
        for line in infile:
            if not line.startswith("T"):
                outfile.write(line)

    out_sim = np.loadtxt("data/out_sim.txt").astype(np.int32)
    out_ref = np.loadtxt("data/out_ref.txt").astype(np.int32)

    if out_sim.shape == out_ref.shape and np.array_equal(out_sim, out_ref):
        print(f"\n\n Success: Outputs match ({out_sim.shape})")
    else:
        print("\n\nError: Output does not match\n")
        print(f"Simulation Output ({out_sim.shape}):\n{out_sim}\n")
        print(f"Expected output ({out_ref.shape}):\n{out_ref}\n")
