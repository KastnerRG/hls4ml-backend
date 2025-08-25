import numpy as np
import os, glob, shutil, subprocess
import math

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

class Layer:
    def emit(self, idx, x_in, y_ref, iterations, layers):
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

    def emit(self, idx, x_in, y_ref, iterations, layers):
        XH, XW, XC = x_in.shape
        YH, YW, YC = y_ref.shape
        KH, KW = self.KH, self.KW
        SH, SW = self.SH, self.SW
        PH = compute_pad(XH, KH, SH, self.padding)
        PW = compute_pad(XW, KW, SW, self.padding)

        # I/O dumps for sim (flat NHWC, 4-wide)
        if idx == 0:
            _save_vec4_i32(f"data/x{idx}.txt", x_in.flatten(), iterations)
        _save_conv_tb4_i32(f"data/a{idx}.txt", y_ref, iterations)

        # model/layer_{idx}.cc
        with open(f"model/layer_{idx}.cc", "a") as f:
            is_prev_conv = idx > 0 and isinstance(layers[idx-1], Conv2D)
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

void f{idx}(input_stream_int32 * __restrict in, output_stream_int32 * __restrict out){{ conv2d_i8_stream(in, out);}}
''')

        # layer_graph wiring
        in_port  = "AIE_IN" if idx == 0 else f"layers[{idx-1}]"
        in_bytes = x_in.size * x_in.itemsize
        with open("model/layer_graph.h", "a") as f:
            f.write(f"layers[{idx}] = kernel::create(f{idx});\n")
            f.write(f'source(layers[{idx}]) = "layer_{idx}.cc";\n')
            f.write(f"auto c{idx} = connect<stream>({in_port}.out[0], layers[{idx}].in[0]);\n")
            f.write(f"fifo_depth(c{idx}) = 64;\n")
            if idx == 0 and in_bytes > 32768:
                f.write(f"single_buffer(layers[{idx}].in[0]);\n")

class Dense(Layer):
    """
    Dense operating row-wise on (R, K) -> (R, N); AIE tiles m=2, k=8, n=8.
    Accepts either 2-D (R,K) or 3-D NHWC (H,W,C). For 3-D, it flattens to (H*W, C).
    """
    def __init__(self, N, shift=0, relu=False, m_tile=2, k_tile=8, n_tile=8):
        self.N = N
        self.shift = shift
        self.relu = relu
        self.m_tile = m_tile
        self.k_tile = k_tile
        self.n_tile = n_tile
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
        assert (R % self.m_tile) == 0 and (K % self.k_tile) == 0 and (self.N % self.n_tile) == 0
        self.K = K
        # random weights (K,N)
        self.W = np.random.randint(0, 128, size=(K, self.N), dtype=np.int8)
        # reference
        y = (x2d.astype(np.int32) @ self.W.astype(np.int32))
        y = (y >> self.shift).astype(np.int8)
        return y

    def emit(self, idx, x_in, y_ref, iterations, layers):
        m, k, n = self.m_tile, self.k_tile, self.n_tile

        # Use the same 2D view we used during forward
        x2d = self._last_in2d if self._last_in2d is not None else self._as_2d(x_in)

        # weights tiled KxN -> k_tiled
        k_tiled = tile_matrix(self.W, k, n)

        # IO files (dense uses tiled dumps)
        x_tiled = tile_matrix(x2d, m, k)
        a_tiled = tile_matrix(y_ref, m, n)
        np.savetxt(f"data/x{idx}.txt", np.tile(x_tiled, (iterations, 1)).reshape(-1, 16), fmt="%s", delimiter=" ")
        np.savetxt(f"data/a{idx}.txt", np.tile(a_tiled, (iterations, 1)).reshape(-1, 16), fmt="%s", delimiter=" ")

        with open(f"model/layer_{idx}.cc", "a") as f:
            f.write(f'''
#define mm_m {m}
#define mm_k {k}
#define mm_n {n}
#define mm_M {x2d.shape[0]}
#define mm_K {x2d.shape[1]}
#define mm_N {self.W.shape[1]}
#define SHIFT {self.shift}
#define DO_RELU {str(self.relu).lower()}

#include <cstdint>
__attribute__((section(".data"))) alignas(32) int8_t matB [{k_tiled.size}] = {{ {", ".join(str(int(x)) for x in k_tiled)} }};

#include "dense_i8_stream.h"

void f{idx}(input_stream_int8 * __restrict in, output_stream_int8 * __restrict out){{ dense_i8(in, out);}}
''')

        # Connect bytes from the *previous* layer as-is (window size is just count of int8)
        in_port   = "AIE_IN" if idx == 0 else f"layers[{idx-1}]"
        num_bytes = x_in.size * x_in.itemsize
        with open("model/layer_graph.h", "a") as f:
            f.write(f"layers[{idx}] = kernel::create(f{idx});\n")
            f.write(f'source(layers[{idx}]) = "layer_{idx}.cc";\n')
            f.write(f"auto c{idx} = connect<stream>({in_port}.out[0], layers[{idx}].in[0]);\n")
            f.write(f"fifo_depth(c{idx}) = 64;\n")
            if idx == 0 and num_bytes > 32768:
                f.write(f"single_buffer(layers[{idx}].in[0]);\n")


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
        for idx, layer in enumerate(self.layers):
            y = layer.forward(x)
            layer.emit(idx, x, y, self.iterations, self.layers)
            x = y

        N_LAYERS = len(self.layers)
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
            f.write(f"auto c{N_LAYERS} = connect<stream>(layers[{N_LAYERS-1}].out[0], AIE_OUT.in[0]);\n")
            f.write(f"fifo_depth(c{N_LAYERS}) = 64;\n")

        # finalize include.h
        with open("model/include.h", "w") as f:
            f.write(f'#define N_LAYERS {N_LAYERS}\n')
            f.write(f'#define ITERATIONS {self.iterations}\n')
            f.write(f'#define TOT_OUT_BYTES {out_bytes*self.iterations}\n')
            f.write(f'#define TOT_IN_BYTES {in_bytes*self.iterations}\n')
            for idx, layer in enumerate(self.layers):
                if isinstance(layer, Conv2D):
                    # conv2d uses int32 input, int8 output
                    f.write(f'void f{idx}(input_stream_int32 * __restrict, output_stream_int32 * __restrict);\n')
                elif isinstance(layer, Dense):
                    # dense uses int8 input, int8 output    
                    f.write(f'void f{idx}(input_stream_int8 * __restrict, output_stream_int8 * __restrict);\n')

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

    # Input (tiny): 4x12x8 NHWC
    XH, XW, XC = 4, 12, 8
    x0 = np.random.randint(0, 128, size=(XH,XW,XC), dtype=np.int8)

    model = Sequential(iterations=iterations)

    model.add(Conv2D(KH=5, KW=7, YC=8, stride=(2,3), padding="same", shift=0, relu=False))
    model.add(Conv2D(KH=3, KW=3, YC=8, stride=(1,1), padding="same", shift=2, relu=False))
    model.add(Conv2D(KH=5, KW=5, YC=8, stride=(1,1), padding="same", shift=2, relu=True))
    # model.add(Dense(N=16, shift=5, relu=False, m_tile=m_tile, k_tile=k_tile, n_tile=n_tile))
    # model.add(Dense(N=16, shift=2, relu=False, m_tile=m_tile, k_tile=k_tile, n_tile=n_tile))
    # model.add(Dense(N=32, shift=3, relu=False, m_tile=m_tile, k_tile=k_tile, n_tile=n_tile))


    # '''
    # Dense only autoencoder
    # '''
    # XH, XW, XC = 2, 128, 1
    # x0 = np.random.randint(0, 128, size=(4,128), dtype=np.int8)
    # model = Sequential(iterations=iterations)
    # model.add(Conv2D(KH=5, KW=7, YC=8, stride=(2,3), padding="same", shift=2, relu=True))
    # model.add(Conv2D(KH=3, KW=3, YC=8, stride=(1,1), padding="same", shift=2, relu=False))
    # model.add(Conv2D(KH=5, KW=5, YC=8, stride=(1,1), padding="same", shift=2, relu=True))
    # model.add(Dense(N=128, shift=5, relu=False, m_tile=m_tile, k_tile=k_tile, n_tile=n_tile))
    # model.add(Dense(N=128, shift=2, relu=False, m_tile=m_tile, k_tile=k_tile, n_tile=n_tile))
    # model.add(Dense(N=128, shift=5, relu=False, m_tile=m_tile, k_tile=k_tile, n_tile=n_tile))
    # model.add(Dense(N= 16, shift=2, relu=False, m_tile=m_tile, k_tile=k_tile, n_tile=n_tile))
    # model.add(Dense(N= 16, shift=5, relu=False, m_tile=m_tile, k_tile=k_tile, n_tile=n_tile))
    # model.add(Dense(N=128, shift=2, relu=False, m_tile=m_tile, k_tile=k_tile, n_tile=n_tile))
    # model.add(Dense(N=128, shift=5, relu=False, m_tile=m_tile, k_tile=k_tile, n_tile=n_tile))
    # model.add(Dense(N=128, shift=2, relu=False, m_tile=m_tile, k_tile=k_tile, n_tile=n_tile))

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
