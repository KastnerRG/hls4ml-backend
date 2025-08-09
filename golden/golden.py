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


# ---------------- Codegen for layers ----------------

def emit_dense(idx, layer, m, k, n, iterations):
    # weights.h
    k_tiled = tile_matrix(layer["k"], k, n)
    array_str = ', '.join(str(x) for x in k_tiled)
    with open("aie/weights.h", 'a') as f:
        f.write(f'__attribute__((section(".data"))) alignas(32) const int8_t k{idx} [{k_tiled.size}] = {{ {array_str} }};\n')

    # IO files
    x_tiled = tile_matrix(layer["x"], m, k)
    a_tiled = tile_matrix(layer["a"], m, n)
    np.savetxt(f"data/x{idx}.txt", np.tile(x_tiled, (iterations, 1)).reshape(-1, 16), fmt="%s", delimiter=" ")
    np.savetxt(f"data/a{idx}.txt", np.tile(a_tiled, (iterations, 1)).reshape(-1, 16), fmt="%s", delimiter=" ")

    # model.cc
    t_m = layer['x'].shape[0] // m
    t_k = layer['x'].shape[1] // k
    t_n = layer['k'].shape[1] // n
    shift = layer['shift']
    is_relu = str(layer['is_relu']).lower()
    with open(f"aie/model_f{idx}.cc", "a") as f:
        f.write('#include "kernels.h"\n#include "weights.h"\n')
        f.write(f"void f{idx}(input_window_int8* __restrict x, output_window_int8 * __restrict a) ")
        f.write(f"{{ dense<{m}, {k}, {n}, {t_m}, {t_k}, {t_n}, {shift}, {is_relu}> (x, a, k{idx}); }}\n")

    # model.h
    with open("aie/model.h", "a") as f:
        f.write(f"void f{idx}( input_window_int8  * __restrict, output_window_int8 * __restrict);\n")

    # layer_graph.h
    num_bytes = layer['x'].size * layer['x'].itemsize
    in_port = "AIE_IN" if idx == 0 else f"layers[{idx-1}]"
    with open("aie/layer_graph.h", "a") as f:
        f.write(f"layers[{idx}] = kernel::create(f{idx});\n")
        f.write(f'source(layers[{idx}]) = "model_f{idx}.cc";\n')
        f.write(f"connect<window<{num_bytes:>5}>>({in_port}.out[0], layers[{idx}].in[0]);\n\n")


def emit_conv2d(idx, layer, params, iterations):
    """
    Emitter for AIE1 vectorized conv2d_v kernel (m=2, n=8, K_TILE=8, stride=1, PAD in {0,1}).

    layer['x'] : NHWC input  (XH,XW,XC) int8
    layer['k'] : weights; accepts either HWIO (KH,KW,XC,YC) or KN (KH*KW*XC, YC)
    layer['a'] : NHWC output (YH,YW,YC) int8  (reference)

    params: dict with keys:
      XH,XW,XC, KH,KW,YC, SH,SW, PH,PW, m(=2), n(=8), SHIFT, is_relu (bool)
    """
    import numpy as np

    XH,XW,XC   = params['XH'], params['XW'], params['XC']
    KH,KW,YC = params['KH'], params['KW'], params['YC']
    SH,SW    = params['SH'], params['SW']
    PAD      = params['PAD'].upper()
    m, n     = params['m'], params['n']
    SHIFT    = params['SHIFT']
    is_relu  = str(params['is_relu']).lower()

    assert m == 2 and n == 8,   "conv2d_v: use m=2, n=8 on AIE1 int8"
    assert XC % 8 == 0 and YC % 8 == 0, "XC and YC must be multiples of 8"

    K_TOTAL = KH * KW * XC

    # ---- Pack weights to [CI8][Tn][KH][KW][8][n] ----
    kernel = layer['k']
    if kernel.shape == (KH, KW, XC, YC):
        W_hwio = kernel
    elif kernel.shape == (K_TOTAL, YC):
        # Interpret as flattened HWIO -> reshape back
        W_hwio = kernel.reshape(KH, KW, XC, YC)
    else:
        raise ValueError(f"weights shape must be ({KH},{KW},{XC},{YC}) or ({K_TOTAL},{YC}), got {kernel.shape}")

    W6 = (W_hwio.reshape(KH,KW,XC//8,8,YC//8,8)).transpose(2,4,0,1,3,5)
    k_tiled = W6.astype(np.int8).ravel()

    # ---- Weights in .data (not PMEM), aligned ----
    with open("aie/weights.h", 'a') as f:
        f.write(
            f'__attribute__((section(".data"))) alignas(32) '
            f'int8_t k{idx} [{k_tiled.size}] = {{ {", ".join(str(int(x)) for x in k_tiled)} }};\n'
        )

    # ---- I/O dumps for sim (keep your style) ----
    if idx == 0:
        x_nhwc = layer['x']
        np.savetxt(
            f"data/x{idx}.txt",
            np.tile(x_nhwc.flatten(), (iterations,1)).reshape(-1,16),
            fmt="%s", delimiter=" "
        )
    a_nhwc = layer['a']
    np.savetxt(
        f"data/a{idx}.txt",
        np.tile(a_nhwc.flatten(), (iterations,1)).reshape(-1,16),
        fmt="%s", delimiter=" "
    )

    # ---- model.cc: instantiate conv2d_v ----
    with open(f"aie/model_f{idx}.cc", "a") as f:
        f.write(
            f'#include "kernels.h"\n#include "weights.h"\n'
            f"void f{idx}(input_window_int8* __restrict x, output_window_int8 * __restrict a) "
            f"{{ conv2d_v_tiny<{XH},{XW},{XC},{YC},{KH},{KW},{SH},{SW},{PAD}>(x, a, k{idx}, {SHIFT}, {is_relu}); }}\n"
        )


    # ---- model.h prototype ----
    with open("aie/model.h", "a") as f:
        f.write(f"void f{idx}( input_window_int8  * __restrict, output_window_int8 * __restrict);\n")

    # ---- layer_graph wiring ----
    in_port  = "AIE_IN" if idx == 0 else f"layers[{idx-1}]"
    in_bytes = layer['x'].size * layer['x'].itemsize
    with open("aie/layer_graph.h", "a") as f:
        f.write(f"layers[{idx}] = kernel::create(f{idx});\n")
        f.write(f'source(layers[{idx}]) = "model_f{idx}.cc";\n')
        f.write(f"connect<window<{in_bytes:>5}>>({in_port}.out[0], layers[{idx}].in[0]);\n\n")



def emit_flatten(idx, XH, XW, XC, iterations):
    # Model function
    with open(f"aie/model_f{idx}.cc", "a") as f:
        f.write('#include "kernels.h"\n#include "weights.h"\n')
        f.write(f"void f{idx}(input_window_int8* __restrict x, output_window_int8 * __restrict a) ")
        f.write(f"{{ flatten_nhwc_to_hw_by_c<{XH},{XW},{XC}>(x, a); }}\n")
    with open("aie/model.h", "a") as f:
        f.write(f"void f{idx}( input_window_int8  * __restrict, output_window_int8 * __restrict);\n")

    # Graph connection
    in_port = "AIE_IN" if idx == 0 else f"layers[{idx-1}]"
    in_bytes = XH*XW*XC
    with open("aie/layer_graph.h", "a") as f:
        f.write(f"layers[{idx}] = kernel::create(f{idx});\n")
        f.write(f'source(layers[{idx}]) = "model_f{idx}.cc";\n')
        f.write(f"connect<window<{in_bytes:>5}>>({in_port}.out[0], layers[{idx}].in[0]);\n\n")


if __name__ == "__main__":
    import numpy as np, os, glob, shutil, subprocess

    # Tiles that AIE1 int8 definitely supports
    m_tile = 2   # spatial pixels per tile
    k_tile = 8   # dense K tile
    n_tile = 8   # channels-per-tile
    iterations = 1

    # Clean
    for path in [
        "data", "aie/include.h", "aie/weights.h", "aie/layer_graph.h", "aie/model*.cc", "aie/model.h",
        "*.log", "aiesimulator_output", "Work", ".Xil",
        ".AIE_SIM_CMD_LINE_OPTIONS", "ISS_RPC_SERVER_PORT",
        "libadf.a", "Map_Report.csv", "pl_sample_counts",
        "plio_throughput_info.json", "sol.db", "aiesim.vcd"
    ]:
        for p in glob.glob(path):
            if os.path.isdir(p): shutil.rmtree(p, ignore_errors=True)
            elif os.path.exists(p): os.remove(p)
    os.makedirs("data", exist_ok=True)

    # Preamble
    open("aie/weights.h","w").close()
    open("aie/model.h","w").close()
    open("aie/layer_graph.h","w").close()

    # ----------------- Network: Conv only -----------------
    # Input (tiny): 4x4x8 NHWC
    XH, XW, XC = 4, 12, 8
    x0 = np.random.randint(0, 128, size=(XH,XW,XC), dtype=np.int8)

    # Conv: 3x3, YC=8, stride=1, pad=1, ReLU (AIE1 conv uses m=2, n=8)
    KH, KW, YC, SH, SW = 5, 7, 8, 2, 3
    SHIFT, IS_RELU = 2, False
    Wc_hwio, Wc_kn = pack_hwio_to_kn(KH, KW, XC, YC)
    y_conv = conv2d_ref(x0, Wc_hwio, stride=(SH,SW), padding="same", shift=SHIFT, relu=IS_RELU)  # -> 4x4x8

    # ----------------- Emit layers -----------------
    layer_idx = 0

    # Conv (m=2, n=8)
    conv_params = dict(
        XH=XH, XW=XW, XC=XC, KH=KH, KW=KW, YC=YC, SH=SH, SW=SW,
        PAD="same", m=2, n=8, SHIFT=SHIFT, is_relu=IS_RELU
    )
    emit_conv2d(layer_idx, {'x': x0, 'k': Wc_kn, 'a': y_conv}, conv_params, iterations)
    layer_idx += 1

    # Model now has only conv
    N_LAYERS = layer_idx
    with open("aie/include.h", "w") as f:
        f.write(f'#define N_LAYERS {N_LAYERS}\n#define ITERATIONS {iterations}')

    # Reference out = conv output flattened in NHWC order (match kernel writeout)
    out_bytes = y_conv.size * y_conv.itemsize
    np.savetxt("data/out_ref.txt",
               np.tile(y_conv.flatten(), (iterations,1)).reshape(-1,16),
               fmt="%s", delimiter=" ")

    # Close graph: connect conv directly to OUT
    with open("aie/layer_graph.h", "a") as f:
        f.write(f"connect<window<{out_bytes:>5}>>(layers[{N_LAYERS-1}].out[0], AIE_OUT.in[0]);\n")

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

