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

def _save_vec4_i32(path, vec_1d: np.ndarray, iterations: int):
    """Pack flat int8 vector into little-endian int32 words (4 bytes per word),
       then write as 4-wide rows. """
    v = np.asarray(vec_1d, dtype=np.int8).ravel()
    pad = (-len(v)) % 4
    if pad:
        v = np.pad(v, (0, pad), mode="constant")
    # pack 4 int8 -> 1 int32 (little endian)
    vv = v.view(np.uint8).reshape(-1, 4)
    words = (vv[:,0].astype(np.uint32)
           | (vv[:,1].astype(np.uint32) << 8)
           | (vv[:,2].astype(np.uint32) << 16)
           | (vv[:,3].astype(np.uint32) << 24)).astype(np.int32)
    arr = np.tile(words, (iterations, 1)).reshape(-1, 4)
    np.savetxt(path, arr, fmt="%d", delimiter=" ")


def _save_conv_tb4_i32(path, y, iterations):
    """Pack TB16 per (yh2,yw,yc8) to 4×int32 (little endian) and save 4-wide."""
    YH, YW, YC = y.shape
    yc8 = YC // 8
    words_i32 = []
    for yh2 in range(0, YH, 2):
        for yw in range(YW):
            for g in range(yc8):
                top = y[yh2, yw, g*8:(g+1)*8].astype(np.int8)
                if yh2+1 < YH:
                    bot = y[yh2+1, yw, g*8:(g+1)*8].astype(np.int8)
                else:
                    bot = np.zeros(8, dtype=np.int8)
                tb16 = np.concatenate([top, bot]).view(np.uint8)
                # pack 16 bytes -> 4 int32 (LE)
                w = (tb16.reshape(4,4)[:,0].astype(np.uint32)
                   | (tb16.reshape(4,4)[:,1].astype(np.uint32) << 8)
                   | (tb16.reshape(4,4)[:,2].astype(np.uint32) << 16)
                   | (tb16.reshape(4,4)[:,3].astype(np.uint32) << 24)).astype(np.int32)
                words_i32.append(w)
    arr = np.tile(np.vstack(words_i32), (iterations, 1))
    np.savetxt(path, arr, fmt="%d", delimiter=" ")


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


# ---------------- Layer classes ----------------

class EmitContext:
    """Tracks unique identifiers for stream connections during graph emit."""
    def __init__(self):
        self._conn_idx = 0

    def new_conn(self):
        name = f"c{self._conn_idx}"
        self._conn_idx += 1
        return name


@dataclass
class KernelSignature:
    idx: int
    inputs: list  # list of dtype strings, e.g., ["int8"]
    outputs: list # list of dtype strings


class Layer:
    def emit(self, layer_idx, kernel_idx, x_in, y_ref, iterations, layers, ctx):
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

    def emit(self, layer_idx, kernel_idx, x_in, y_ref, iterations, layers, ctx):
        XH, XW, XC = x_in.shape
        YH, YW, YC = y_ref.shape
        KH, KW = self.KH, self.KW
        SH, SW = self.SH, self.SW
        PH = compute_pad(XH, KH, SH, self.padding)
        PW = compute_pad(XW, KW, SW, self.padding)

        # I/O dumps for sim (flat NHWC, 4-wide)
        if kernel_idx == 0:
            _save_vec4_i32(f"data/x{kernel_idx}.txt", x_in.flatten(), iterations)
        _save_conv_tb4_i32(f"data/a{kernel_idx}.txt", y_ref, iterations)

        # model/layer_{idx}.cc
        with open(f"model/layer_{kernel_idx}.cc", "a") as f:
            is_prev_conv = layer_idx > 0 and isinstance(layers[layer_idx-1], Conv2D)
            f.write(f"#define IN_TB16 {1 if is_prev_conv else 0}\n")
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

#include "conv2d_i8_stream.h"

void f{kernel_idx}(input_stream_int32 * __restrict in, output_stream_int32 * __restrict out){{ conv2d_i8_stream(in, out);}}
''')

        # layer_graph wiring
        in_port  = "AIE_IN" if kernel_idx == 0 else f"layers[{kernel_idx-1}]"
        in_bytes = x_in.size * x_in.itemsize
        with open("model/layer_graph.h", "a") as f:
            f.write(f"layers[{kernel_idx}] = kernel::create(f{kernel_idx});\n")
            f.write(f'source(layers[{kernel_idx}]) = "layer_{kernel_idx}.cc";\n')
            conn = ctx.new_conn()
            f.write(f"auto {conn} = connect<stream>({in_port}.out[0], layers[{kernel_idx}].in[0]);\n")
            f.write(f"fifo_depth({conn}) = 64;\n")
            if kernel_idx == 0 and in_bytes > 32768:
                f.write(f"single_buffer(layers[{kernel_idx}].in[0]);\n")

        sig = KernelSignature(idx=kernel_idx, inputs=["int32"], outputs=["int32"])
        return kernel_idx + 1, [sig]

class Dense(Layer):
    """
    Dense operating row-wise on (R, K) -> (R, N); AIE tiles m=2, k=8, n=8.
    Accepts either 2-D (R,K) or 3-D NHWC (H,W,C). For 3-D, it flattens to (H*W, C).
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
        assert (R % self.m_tile) == 0 and (K % self.k_tile) == 0 and (self.N % (self.n_tile * self.o_tiles)) == 0
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

    def emit(self, layer_idx, kernel_idx, x_in, y_ref, iterations, layers, ctx):
        m, k, n = self.m_tile, self.k_tile, self.n_tile

        # Use the same 2D view we used during forward
        x2d = self._last_in2d if self._last_in2d is not None else self._as_2d(x_in)
        rows, cols = x2d.shape

        # IO dumps (dense uses tiled dumps). Associate input dump with the first kernel in this block.
        x_tiled = tile_matrix(x2d, m, k)
        np.savetxt(
            f"data/x{kernel_idx}.txt",
            np.tile(x_tiled, (iterations, 1)).reshape(-1, 16),
            fmt="%s",
            delimiter=" "
        )

        specs = []

        def _write_kernel(idx, body):
            with open(f"model/layer_{idx}.cc", "a") as f:
                f.write(body)

        def _emit_kernel_decl(idx, source_name):
            with open("model/layer_graph.h", "a") as f:
                f.write(f"layers[{idx}] = kernel::create(f{idx});\n")
                f.write(f'source(layers[{idx}]) = "{source_name}";\n')

        def _connect(src_expr, dst_idx, dst_port=0, allow_single_buffer=False):
            conn = ctx.new_conn()
            with open("model/layer_graph.h", "a") as f:
                f.write(f"auto {conn} = connect<stream>({src_expr}, layers[{dst_idx}].in[{dst_port}]);\n")
                f.write(f"fifo_depth({conn}) = 64;\n")
                if allow_single_buffer:
                    num_bytes = x_in.size * x_in.itemsize
                    if num_bytes > 32768:
                        f.write(f"single_buffer(layers[{dst_idx}].in[{dst_port}]);\n")

        def _upstream_src():
            return "AIE_IN.out[0]" if kernel_idx == 0 else f"layers[{kernel_idx-1}].out[0]"

        if self.o_tiles == 1:
            k_tiled = tile_matrix(self.W, k, n)
            a_tiled = tile_matrix(y_ref, m, n)
            np.savetxt(
                f"data/a{kernel_idx}.txt",
                np.tile(a_tiled, (iterations, 1)).reshape(-1, 16),
                fmt="%s",
                delimiter=" "
            )

            body = f'''
#define mm_m {m}
#define mm_k {k}
#define mm_n {n}
#define mm_M {rows}
#define mm_K {cols}
#define mm_N {self.W.shape[1]}
#define SHIFT {self.shift}
#define DO_RELU {str(self.relu).lower()}

#include <cstdint>
__attribute__((section(".data"))) alignas(32) int8_t matB [{k_tiled.size}] = {{ {", ".join(str(int(x)) for x in k_tiled)} }};

#include "dense_i8_stream.h"

void f{kernel_idx}(input_stream_int8 * __restrict in, output_stream_int8 * __restrict out){{ dense_i8(in, out);}}
'''
            _write_kernel(kernel_idx, body)
            _emit_kernel_decl(kernel_idx, f"layer_{kernel_idx}.cc")
            _connect(_upstream_src(), kernel_idx, allow_single_buffer=(kernel_idx == 0))

            specs.append(KernelSignature(idx=kernel_idx, inputs=["int8"], outputs=["int8"]))
            return kernel_idx + 1, specs

        assert (self.o_tiles & (self.o_tiles - 1)) == 0, "o_tiles must be a power of two for balanced splitting"

        a_tiled = tile_matrix(y_ref, m, n)

        next_idx = kernel_idx
        upstream = _upstream_src()

        def build_broadcast(count, src_expr, first):
            nonlocal next_idx
            if count == 1:
                return [src_expr]
            fan = 2
            b_idx = next_idx
            next_idx += 1
            body = f'''
#define LB_IN_M {rows}
#define LB_IN_K {cols}
#define LB_TILE_M {m}
#define LB_TILE_K {k}
#define O_TILES {fan}

#include "dense_input_broadcast.h"

{self._emit_broadcast_signature(b_idx, fan)}
'''
            _write_kernel(b_idx, body)
            _emit_kernel_decl(b_idx, f"layer_{b_idx}.cc")
            specs.append(KernelSignature(idx=b_idx, inputs=["int8"], outputs=["int8"] * fan))
            _connect(src_expr, b_idx, allow_single_buffer=first)
            child_sources = []
            child_count = count // fan
            for port in range(fan):
                child_src = f"layers[{b_idx}].out[{port}]"
                child_sources.extend(build_broadcast(child_count, child_src, False))
            return child_sources

        leaf_sources = build_broadcast(self.o_tiles, upstream, True)

        tile_outputs = []
        for tile, src in enumerate(leaf_sources):
            tile_idx = next_idx
            next_idx += 1
            n_begin = tile * self.N_per_tile
            n_end = (tile + 1) * self.N_per_tile
            W_slice = self.W[:, n_begin:n_end]
            k_tiled_slice = tile_matrix(W_slice, k, n)
            body = f'''
#define mm_m {m}
#define mm_k {k}
#define mm_n {n}
#define mm_M {rows}
#define mm_K {cols}
#define mm_N {self.N_per_tile}
#define SHIFT {self.shift}
#define DO_RELU {str(self.relu).lower()}

#include <cstdint>
__attribute__((section(".data"))) alignas(32) int8_t matB [{k_tiled_slice.size}] = {{ {", ".join(str(int(x)) for x in k_tiled_slice)} }};

#include "dense_i8_stream.h"

void f{tile_idx}(input_stream_int8 * __restrict in, output_stream_int8 * __restrict out){{ dense_i8(in, out);}}
'''
            _write_kernel(tile_idx, body)
            _emit_kernel_decl(tile_idx, f"layer_{tile_idx}.cc")
            specs.append(KernelSignature(idx=tile_idx, inputs=["int8"], outputs=["int8"]))
            _connect(src, tile_idx)
            tile_outputs.append(f"layers[{tile_idx}].out[0]")

        final_idx = None

        def build_concat(start, count):
            nonlocal next_idx, final_idx
            if count == 1:
                return tile_outputs[start], None
            half = count // 2
            left_src, _ = build_concat(start, half)
            right_src, _ = build_concat(start + half, half)
            c_idx = next_idx
            next_idx += 1
            body = f'''
#define LB_OUT_M {rows}
#define LB_OUT_N {count * self.N_per_tile}
#define LB_TILE_M {m}
#define LB_TILE_N {n}
#define O_TILES 2

#include "dense_output_concat.h"

{self._emit_concat_signature(c_idx, 2)}
'''
            _write_kernel(c_idx, body)
            _emit_kernel_decl(c_idx, f"layer_{c_idx}.cc")
            specs.append(KernelSignature(idx=c_idx, inputs=["int8", "int8"], outputs=["int8"]))
            _connect(left_src, c_idx, dst_port=0)
            _connect(right_src, c_idx, dst_port=1)
            final_idx = c_idx
            return f"layers[{c_idx}].out[0]", c_idx

        final_src, _ = build_concat(0, self.o_tiles)
        assert final_idx is not None

        np.savetxt(
            f"data/a{final_idx}.txt",
            np.tile(a_tiled, (iterations, 1)).reshape(-1, 16),
            fmt="%s",
            delimiter=" "
        )

        return next_idx, specs

    def _emit_broadcast_signature(self, func_idx, fan):
        out_params = [f"output_stream_int8 * __restrict out{t}" for t in range(fan)]
        params = ["input_stream_int8 * __restrict in"] + out_params
        outs_array = ", ".join([f"out{t}" for t in range(fan)])
        return (
            f"void f{func_idx}({', '.join(params)})\n{{\n"
            f"  output_stream_int8 * outs[O_TILES] = {{ {outs_array} }};\n"
            f"  dense_input_broadcast(in, outs);\n}}\n"
        )

    def _emit_concat_signature(self, func_idx, fan):
        in_params = [f"input_stream_int8 * __restrict in{t}" for t in range(fan)]
        params = in_params + ["output_stream_int8 * __restrict out"]
        ins_array = ", ".join([f"in{t}" for t in range(fan)])
        return (
            f"void f{func_idx}({', '.join(params)})\n{{\n"
            f"  input_stream_int8 * ins[O_TILES] = {{ {ins_array} }};\n"
            f"  dense_output_concat(ins, out);\n}}\n"
        )


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
        # ensure clean dirs already set up by caller; we only write files here
        # also create empty layer_graph for appends
        open("model/layer_graph.h", "w").close()

        x = x0
        kernel_idx = 0
        ctx = EmitContext()
        kernel_signatures = []
        for layer_idx, layer in enumerate(self.layers):
            y = layer.forward(x)
            kernel_idx, specs = layer.emit(layer_idx, kernel_idx, x, y, self.iterations, self.layers, ctx)
            kernel_signatures.extend(specs)
            x = y

        N_LAYERS = kernel_idx
        in_bytes = x0.size * x0.itemsize

        # last layer output file for final compare (depends on layer type)
        last = self.layers[-1]
        if isinstance(last, Dense):
            # dense writes in tiled (m=2, n=8) layout at the output edge
            m_tile, n_tile = last.m_tile, last.n_tile
            tiled_last = tile_matrix(x, m_tile, n_tile)
            np.savetxt("data/out_ref.txt",
                       np.tile(tiled_last, (self.iterations,1)).reshape(-1,16),
                       fmt="%s", delimiter=" ")
            out_bytes = x.size * x.itemsize
        else:
            # conv/flatten raw bytes
            _save_conv_tb4_i32("data/out_ref.txt", x, self.iterations)
            out_bytes = x.size * x.itemsize

        with open("model/layer_graph.h", "a") as f:
            if out_bytes >= 32768:
                f.write(f"single_buffer(layers[{N_LAYERS-1}].out[0]);\n")
            conn = ctx.new_conn()
            f.write(f"auto {conn} = connect<stream>(layers[{N_LAYERS-1}].out[0], AIE_OUT.in[0]);\n")
            f.write(f"fifo_depth({conn}) = 64;\n")

        # finalize include.h
        with open("model/include.h", "w") as f:
            f.write(f'#define N_LAYERS {N_LAYERS}\n')
            f.write(f'#define ITERATIONS {self.iterations}\n')
            f.write(f'#define TOT_OUT_BYTES {out_bytes*self.iterations}\n')
            f.write(f'#define TOT_IN_BYTES {in_bytes*self.iterations}\n')
            for sig in sorted(kernel_signatures, key=lambda s: s.idx):
                params = []
                for dtype in sig.inputs:
                    params.append(f"input_stream_{dtype} * __restrict")
                for dtype in sig.outputs:
                    params.append(f"output_stream_{dtype} * __restrict")
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
    model.add(Dense(N=128, shift=5, relu=False, m_tile=m_tile, k_tile=k_tile, n_tile=n_tile, o_tiles=4))
    model.add(Dense(N= 16, shift=2, relu=False, m_tile=m_tile, k_tile=k_tile, n_tile=n_tile))



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
        print(f"\n\n Success: Outputs match ({out_sim.shape})")
    else:
        print("\n\nError: Output does not match\n")
        print(f"Simulation Output ({out_sim.shape}):\n{out_sim}\n")
        print(f"Expected output ({out_ref.shape}):\n{out_ref}\n")
