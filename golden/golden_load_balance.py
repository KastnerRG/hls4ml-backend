import numpy as np
import os, glob, shutil, subprocess
import math
from dataclasses import dataclass

# ---------------- Common helpers ----------------

def tile_matrix(matrix, row_tiles, col_tiles):  # (R,C) -> (R/r, C/c, r, c).flatten()
    rows, cols = matrix.shape
    assert rows % row_tiles == 0 and cols % col_tiles == 0, "Matrix must be divisible by block sizes"
    reshaped = matrix.reshape(rows // row_tiles, row_tiles, cols // col_tiles, col_tiles)
    transposed = reshaped.transpose(0, 2, 1, 3)  # (R/r, C/c, r, c)
    return transposed.flatten()

def _save_vec16(path, vec_1d: np.ndarray, iterations: int):
    """Save an int8 vector as 16-wide rows, padding to multiple of 16 if needed."""
    vec = vec_1d.reshape(-1)
    pad = (-len(vec)) % 16
    if pad:
        vec = np.pad(vec, (0, pad), mode="constant")
    np.savetxt(path, np.tile(vec, (iterations, 1)).reshape(-1, 16), fmt="%s", delimiter=" ")

def pack_hwio_to_kn(KH, KW, XC, YC):
    # HWIO -> [k, YC], where k = KH*KW*XC. kk = ((kh*KW)+kw)*XC + ci
    k = KH * KW * XC
    W2 = np.random.randint(0, 128, size=(KH, KW, XC, YC), dtype=np.int8)
    W_kn = np.zeros((k, YC), dtype=np.int8)
    idx = 0
    for kh in range(KH):
        for kw in range(KW):
            for ci in range(XC):
                W_kn[idx, :] = W2[kh, kw, ci, :]
                idx += 1
    return W2, W_kn

def compute_pad(in_size, k, stride, mode):
    if mode == "valid":
        return 0
    elif mode == "same":
        out_size = math.ceil(in_size / stride)
        total_pad = max((out_size - 1) * stride + k - in_size, 0)
        return total_pad // 2  # symmetric padding
    else:
        raise ValueError(f"Invalid padding mode: {mode}")

def conv2d_ref(x_nhwc, W_hwio, stride=(1,1), padding="same", shift=0, relu=False):
    # N=1 only; NHWC * HWIO -> NHWC (int8 accum shift)
    XH, XW, XC = x_nhwc.shape
    KH, KW, CI2, YC = W_hwio.shape
    assert XC == CI2

    SH, SW = stride
    PH = compute_pad(XH, KH, SH, padding)
    PW = compute_pad(XW, KW, SW, padding)

    YH = (XH + 2*PH - KH)//SH + 1
    YW = (XW + 2*PW - KW)//SW + 1

    y = np.zeros((YH, YW, YC), dtype=np.int32)
    for oh in range(YH):
        for ow in range(YW):
            acc = np.zeros((YC,), dtype=np.int32)
            for kh in range(KH):
                ih = oh*SH + kh - PH
                if ih < 0 or ih >= XH:
                    continue
                for kw in range(KW):
                    iw = ow*SW + kw - PW
                    if iw < 0 or iw >= XW:
                        continue
                    xi = x_nhwc[ih, iw, :].astype(np.int32)     # input vector
                    w  = W_hwio[kh, kw, :, :].astype(np.int32) # weights
                    acc += xi @ w
            y[oh, ow, :] = acc

    y = (y >> shift).astype(np.int8)
    if relu:
        y = np.maximum(0, y)
    return y


class EmitContext:
    """Utility to build layer_graph.h incrementally."""
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

    def connect_window(self, src_expr, dst_expr, num_bytes, allow_single_buffer=False):
        conn = self.new_conn()
        self._write(f"auto {conn} = connect<window<{num_bytes}>>({src_expr}, {dst_expr});")
        if allow_single_buffer:
            self._write(f"single_buffer({dst_expr});")
        return conn

    def create_shared_buffer(self, num_bytes, num_inputs, num_outputs, prefix="sb"):
        name = f"{prefix}{self._sb_idx}"
        self._sb_idx += 1
        self._write(f"auto {name} = adf::shared_buffer<int8>::create({{{num_bytes}}}, {num_inputs}, {num_outputs});")
        tp = f"{name}_tp"
        ap = f"{name}_ap"
        self._write(f"adf::tiling_parameters {tp};")
        self._write(f"{tp}.buffer_dimension = {{{num_bytes}}};")
        self._write(f"{tp}.tiling_dimension = {{{num_bytes}}};")
        self._write(f"{tp}.offset = {{0}};")
        self._write(f"{tp}.tile_traversal.push_back(adf::traversing_parameters{{0, {num_bytes}, 1}});")
        self._write(f"adf::access_pattern {ap}({tp});")
        for i in range(num_inputs):
            self._write(f"adf::write_access({name}.in[{i}]) = {{ {ap} }};")
        for o in range(num_outputs):
            self._write(f"adf::read_access({name}.out[{o}]) = {{ {ap} }};")
        return name


@dataclass
class KernelSignature:
    idx: int
    inputs: list  # e.g., ["window_int8", "window_int8"]
    outputs: list


# ---------------- Layer classes ----------------

class Layer:
    def emit(self, layer_idx, kernel_idx, src_expr, src_bytes, x_in, y_ref, iterations, ctx):
        raise NotImplementedError
    def forward(self, x_in):
        raise NotImplementedError

class Conv2D(Layer):
    """
    Conv2D(XC multiple of 8, YC multiple of 8). Emits a macro-parameterized AIE conv kernel.
    padding: "same" or "valid"
    """
    def __init__(self, KH, KW, YC, stride=(1,1), padding="same", shift=0, relu=False):
        self.KH, self.KW = KH, KW
        self.YC = YC
        self.SH, self.SW = stride
        self.padding = padding
        self.shift = shift
        self.relu = relu
        # weights allocated at build time when input XC is known

    def _pack_weights(self, W_hwio):
        KH, KW, XC, YC = W_hwio.shape
        W6 = (W_hwio.reshape(KH,KW,XC//8,8,YC//8,8)).transpose(2,4,0,1,3,5)
        return W6.astype(np.int8).ravel()

    def forward(self, x_in):
        XH, XW, XC = x_in.shape
        assert XC % 8 == 0 and self.YC % 8 == 0
        # init weights (random) for this layer given XC
        W_hwio, W_kn = pack_hwio_to_kn(self.KH, self.KW, XC, self.YC)
        self.W_hwio = W_hwio
        self.k_tiled = self._pack_weights(W_hwio)
        y = conv2d_ref(
            x_in, W_hwio,
            stride=(self.SH,self.SW),
            padding=self.padding,
            shift=self.shift,
            relu=self.relu
        )
        return y

    def emit(self, layer_idx, kernel_idx, src_expr, src_bytes, x_in, y_ref, iterations, ctx):
        XH, XW, XC = x_in.shape
        YH, YW, YC = y_ref.shape
        KH, KW = self.KH, self.KW
        SH, SW = self.SH, self.SW
        PH = compute_pad(XH, KH, SH, self.padding)
        PW = compute_pad(XW, KW, SW, self.padding)

        # I/O dumps for sim (flat NHWC, 16-wide)
        if layer_idx == 0:
            _save_vec16(f"data/x{layer_idx}.txt", x_in.flatten(), iterations)
        _save_vec16(f"data/a{layer_idx}.txt", y_ref.flatten(), iterations)

        # model/layer_{kernel_idx}.cc
        with open(f"model/layer_{kernel_idx}.cc", "a") as f:
            f.write(f'''
#define XH {XH}
#define XW {XW}
#define XC {XC}
#define YH {YH}
#define YW {YW}
#define YC {YC}
#define KH {KH}
#define KW {KW}
#define SH {SH}
#define SW {SW}
#define PH {PH}
#define PW {PW}
#define SHIFT {self.shift}
#define DO_RELU {str(self.relu).lower()}

#include <cstdint>
__attribute__((section(".data"))) alignas(32) int8_t k_p [{self.k_tiled.size}] = {{ {", ".join(str(int(x)) for x in self.k_tiled)} }};

#include "conv2d_i8.h"

void f{kernel_idx}(input_window_int8 * __restrict in, output_window_int8 * __restrict out){{ conv2d_i8(in, out);}}
''')

        in_bytes = x_in.size * x_in.itemsize
        ctx._write(f"layers[{kernel_idx}] = kernel::create(f{kernel_idx});")
        ctx._write(f'source(layers[{kernel_idx}]) = "layer_{kernel_idx}.cc";')
        ctx.connect_window(src_expr, f"layers[{kernel_idx}].in[0]", in_bytes,
                           allow_single_buffer=(src_expr == "AIE_IN.out[0]" and in_bytes > 32768))

        sig = KernelSignature(idx=kernel_idx,
                              inputs=["window_int8"],
                              outputs=["window_int8"])
        ctx.kernel_sigs.append(sig)

        out_expr = f"layers[{kernel_idx}].out[0]"
        out_bytes = y_ref.size * y_ref.itemsize
        return kernel_idx + 1, out_expr, out_bytes

class Dense(Layer):
    """
    Dense operating row-wise on (R, K) -> (R, N); AIE tiles m=2, k=8, n=8.
    Supports optional output tiling (o_tiles) using shared buffers + concat kernels.
    """
    def __init__(self, N, shift=0, relu=False, m_tile=2, k_tile=8, n_tile=8, o_tiles=1):
        self.N = N
        self.shift = shift
        self.relu = relu
        self.m_tile = m_tile
        self.k_tile = k_tile
        self.n_tile = n_tile
        self.o_tiles = o_tiles
        self._last_in2d = None  # cached 2D view used in emit

    def _as_2d(self, x_in: np.ndarray):
        if x_in.ndim == 2:
            return x_in
        elif x_in.ndim == 3:
            XH, XW, XC = x_in.shape
            return x_in.reshape(XH * XW, XC)
        else:
            raise ValueError(f"Dense expects 2D or 3D input, got shape {x_in.shape}")

    def forward(self, x_in):
        x2d = self._as_2d(x_in)
        self._last_in2d = x2d  # remember for emit()
        R, K = x2d.shape
        assert (R % self.m_tile) == 0 and (K % self.k_tile) == 0
        assert (self.N % (self.n_tile * self.o_tiles)) == 0
        assert self.o_tiles >= 1 and (self.o_tiles & (self.o_tiles - 1)) == 0
        self.K = K
        self.N_per_tile = self.N // self.o_tiles
        # random weights (K,N)
        self.W = np.random.randint(0, 128, size=(K, self.N), dtype=np.int8)
        # reference
        y = (x2d.astype(np.int32) @ self.W.astype(np.int32))
        y = (y >> self.shift).astype(np.int8)
        if self.relu:
            y = np.maximum(0, y)
        return y

    def emit(self, layer_idx, kernel_idx, src_expr, src_bytes, x_in, y_ref, iterations, ctx):
        m, k, n = self.m_tile, self.k_tile, self.n_tile
        x2d = self._last_in2d if self._last_in2d is not None else self._as_2d(x_in)

        # IO dumps (dense uses tiled dumps)
        x_tiled = tile_matrix(x2d, m, k)
        a_tiled = tile_matrix(y_ref, m, n)
        np.savetxt(f"data/x{layer_idx}.txt", np.tile(x_tiled, (iterations, 1)).reshape(-1, 16), fmt="%s", delimiter=" ")
        np.savetxt(f"data/a{layer_idx}.txt", np.tile(a_tiled, (iterations, 1)).reshape(-1, 16), fmt="%s", delimiter=" ")

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
#include <adf.h>
__attribute__((section(".data"))) alignas(32) int8_t matB [{k_tiled.size}] = {{ {", ".join(str(int(x)) for x in k_tiled)} }};

#include "dense_i8.h"

void f{idx}(input_window_int8 * __restrict in, output_window_int8 * __restrict out){{ dense_i8(in, out);}}
''')
            ctx._write(f"layers[{idx}] = kernel::create(f{idx});")
            ctx._write(f'source(layers[{idx}]) = "layer_{idx}.cc";')
            ctx.kernel_sigs.append(KernelSignature(idx=idx,
                                                  inputs=["window_int8"],
                                                  outputs=["window_int8"]))

        def emit_concat_kernel(idx, left_bytes, right_bytes):
            with open(f"model/layer_{idx}.cc", "a") as f:
                f.write(f'''
#define LEFT_BYTES {left_bytes}
#define RIGHT_BYTES {right_bytes}

#include <cstdint>
#include <adf.h>

void f{idx}(input_window_int8 * __restrict in0,
            input_window_int8 * __restrict in1,
            output_window_int8 * __restrict out){{
  const int8_t * __restrict p0 = (const int8_t*)in0->ptr;
  const int8_t * __restrict p1 = (const int8_t*)in1->ptr;
  int8_t * __restrict po = (int8_t*)out->ptr;
  for (int i = 0; i < LEFT_BYTES; ++i) *po++ = *p0++;
  for (int i = 0; i < RIGHT_BYTES; ++i) *po++ = *p1++;
}}
''')
            ctx._write(f"layers[{idx}] = kernel::create(f{idx});")
            ctx._write(f'source(layers[{idx}]) = "layer_{idx}.cc";')
            ctx.kernel_sigs.append(KernelSignature(idx=idx,
                                                  inputs=["window_int8", "window_int8"],
                                                  outputs=["window_int8"]))

        in_bytes = x_in.size * x_in.itemsize
        allow_single = (src_expr == "AIE_IN.out[0]" and in_bytes > 32768)

        if self.o_tiles == 1:
            emit_dense_kernel(kernel_idx, self.W, self.N)
            ctx.connect_window(src_expr, f"layers[{kernel_idx}].in[0]", in_bytes, allow_single)
            out_expr = f"layers[{kernel_idx}].out[0]"
            out_bytes = y_ref.size * y_ref.itemsize
            return kernel_idx + 1, out_expr, out_bytes

        sb_in = ctx.create_shared_buffer(in_bytes, 1, self.o_tiles, prefix=f"dense_in_{layer_idx}_")
        ctx.connect_window(src_expr, f"{sb_in}.in[0]", in_bytes, allow_single)

        branch_outputs = []
        bytes_per_branch = x2d.shape[0] * self.N_per_tile  # int8 bytes
        for tile in range(self.o_tiles):
            tile_idx = kernel_idx
            kernel_idx += 1
            n_begin = tile * self.N_per_tile
            n_end = (tile + 1) * self.N_per_tile
            W_slice = self.W[:, n_begin:n_end]
            emit_dense_kernel(tile_idx, W_slice, self.N_per_tile)
            ctx.connect_window(f"{sb_in}.out[{tile}]", f"layers[{tile_idx}].in[0]", in_bytes)
            branch_outputs.append(f"layers[{tile_idx}].out[0]")

        row_tiles = x2d.shape[0] // m
        chunk_bytes = m * self.N_per_tile
        concat_idx = kernel_idx
        kernel_idx += 1

        inputs_sig = []
        param_list = []
        for t in range(self.o_tiles):
            inputs_sig.append("window_int8")
            param_list.append(f"input_window_int8 * __restrict in{t}")
        param_list.append("output_window_int8 * __restrict out")
        with open(f"model/layer_{concat_idx}.cc", "a") as f:
            f.write(f'''
#define ROW_TILES {row_tiles}
#define CHUNK_BYTES {chunk_bytes}
#define O_TILES {self.o_tiles}

#include <cstdint>
#include <adf.h>

void f{concat_idx}({', '.join(param_list)}){{
  const int8_t * __restrict in_ptr[O_TILES] = {{ {', '.join(f'(const int8_t*)in{t}->ptr' for t in range(self.o_tiles))} }};
  int8_t * __restrict out_ptr = (int8_t*)out->ptr;
  for (int rt = 0; rt < ROW_TILES; ++rt){{
    for (int t = 0; t < O_TILES; ++t){{
      const int8_t * __restrict src = in_ptr[t];
      for (int i = 0; i < CHUNK_BYTES; ++i){{
        *out_ptr++ = *src++;
      }}
      in_ptr[t] = src;
    }}
  }}
}}
''')

        ctx._write(f"layers[{concat_idx}] = kernel::create(f{concat_idx});")
        ctx._write(f'source(layers[{concat_idx}]) = "layer_{concat_idx}.cc";')
        ctx.kernel_sigs.append(KernelSignature(idx=concat_idx,
                                              inputs=inputs_sig,
                                              outputs=["window_int8"]))
        for t, expr in enumerate(branch_outputs):
            ctx.connect_window(expr, f"layers[{concat_idx}].in[{t}]", bytes_per_branch)

        out_expr = f"layers[{concat_idx}].out[0]"
        out_bytes = bytes_per_branch * self.o_tiles
        return kernel_idx, out_expr, out_bytes


# ---------------- Sequential-ish model ----------------

class Sequential:
    def __init__(self, iterations=1):
        self.layers = []
        self.iterations = iterations

    def add(self, layer: Layer):
        self.layers.append(layer)

    def build_and_emit(self, x0: np.ndarray):
        """
        Runs reference forward pass, emits per-layer .cc and layer_graph wiring,
        and returns final reference output.
        """
        graph_path = "model/layer_graph.h"
        open(graph_path, "w").close()
        ctx = EmitContext(graph_path)

        x = x0
        kernel_idx = 0
        src_expr = "AIE_IN.out[0]"
        src_bytes = x0.size * x0.itemsize
        for layer_idx, layer in enumerate(self.layers):
            y = layer.forward(x)
            kernel_idx, src_expr, src_bytes = layer.emit(
                layer_idx, kernel_idx, src_expr, src_bytes, x, y, self.iterations, ctx
            )
            x = y

        N_LAYERS = kernel_idx
        in_bytes = x0.size * x0.itemsize

        # last layer output file for final compare (depends on layer type)
        last = self.layers[-1]
        if isinstance(last, Dense):
            m_tile, n_tile = last.m_tile, last.n_tile
            tiled_last = tile_matrix(x, m_tile, n_tile)
            np.savetxt("data/out_ref.txt",
                       np.tile(tiled_last, (self.iterations,1)).reshape(-1,16),
                       fmt="%s", delimiter=" ")
            out_bytes = x.size * x.itemsize
        else:
            _save_vec16("data/out_ref.txt", x.flatten(), self.iterations)
            out_bytes = x.size * x.itemsize

        ctx.connect_window(src_expr, "AIE_OUT.in[0]", out_bytes)

        # finalize include.h
        with open("model/include.h", "w") as f:
            f.write(f'#define N_LAYERS {N_LAYERS}\n')
            f.write(f'#define ITERATIONS {self.iterations}\n')
            f.write(f'#define TOT_OUT_BYTES {out_bytes*self.iterations}\n')
            f.write(f'#define TOT_IN_BYTES {in_bytes*self.iterations}\n')
            for sig in sorted(ctx.kernel_sigs, key=lambda s: s.idx):
                params = []
                for spec in sig.inputs:
                    params.append("input_window_int8 * __restrict" if spec == "window_int8" else "input_window_int8 * __restrict")
                for spec in sig.outputs:
                    params.append("output_window_int8 * __restrict" if spec == "window_int8" else "output_window_int8 * __restrict")
                param_str = ", ".join(params)
                f.write(f"void f{sig.idx}({param_str});\n")

        return x


# ---------------- Main ----------------

if __name__ == "__main__":
    # Tiles that AIE1 int8 definitely supports
    m_tile = 2   # spatial pixels per tile
    k_tile = 8   # dense K tile
    n_tile = 8   # channels-per-tile
    iterations = 1

    # Clean
    for path in [
        "data", "model",
        "*.log", "aiesimulator_output", "Work", ".Xil",
        ".AIE_SIM_CMD_LINE_OPTIONS", "ISS_RPC_SERVER_PORT",
        "libadf.a", "Map_Report.csv", "pl_sample_counts",
        "plio_throughput_info.json", "sol.db", "aiesim.vcd"
    ]:
        for p in glob.glob(path):
            if os.path.isdir(p): shutil.rmtree(p, ignore_errors=True)
            elif os.path.exists(p): os.remove(p)
    os.makedirs("data", exist_ok=True)
    os.makedirs("model", exist_ok=True)

    BATCH=4
    INPUTS=128
    OUTPUTS=128

    x0 = np.random.randint(0, 128, size=(BATCH,INPUTS), dtype=np.int8)
    model = Sequential(iterations=iterations)
    model.add(Dense(N=128, shift=5, relu=False, m_tile=m_tile, k_tile=k_tile, n_tile=n_tile, o_tiles=1))
    model.add(Dense(N= 16, shift=2, relu=False, m_tile=m_tile, k_tile=k_tile, n_tile=n_tile, o_tiles=1))

    # Build, emit code, and get reference
    y_ref_final = model.build_and_emit(x0)

    # Build & sim
    subprocess.run(["./run.sh"], check=True)

    # Verify
    aie_out_path = "aiesimulator_output/data/out_sim.txt"
    assert os.path.exists(aie_out_path), f"Error: Output file {aie_out_path} does not exist."
    with open(aie_out_path, "r") as infile, open("data/out_sim.txt", "w") as outfile:
        for line in infile:
            if not line.startswith("T"):
                outfile.write(line)

    out_sim = np.loadtxt("data/out_sim.txt").astype(np.int32)
    out_ref = np.loadtxt("data/out_ref.txt").astype(np.int32)

    if out_sim.shape == out_ref.shape and np.array_equal(out_sim, out_ref):
        print(f"\n\n Success: Outputs match ({out_sim.shape})\n\n{out_sim}\n\n")
    else:
        print("\n\nError: Output does not match\n")
        print(f"Simulation Output ({out_sim.shape}):\n{out_sim}\n")
        print(f"Expected output ({out_ref.shape}):\n{out_ref}\n")
