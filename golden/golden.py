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

    k_tiled = tile_matrix(layer["k"], k, n)


    # IO files
    x_tiled = tile_matrix(layer["x"], m, k)
    a_tiled = tile_matrix(layer["a"], m, n)
    np.savetxt(f"data/x{idx}.txt", np.tile(x_tiled, (iterations, 1)).reshape(-1, 16), fmt="%s", delimiter=" ")
    np.savetxt(f"data/a{idx}.txt", np.tile(a_tiled, (iterations, 1)).reshape(-1, 16), fmt="%s", delimiter=" ")

    # model.cc
    t_m = layer['x'].shape[0] // m
    t_k = layer['x'].shape[1] // k
    t_n = layer['k'].shape[1] // n
    with open(f"model/layer_{idx}.cc", "a") as f:
        f.write(f'''
#define m_api {m}
#define k_api {k}
#define n_api {n}
#define Tm {t_m}
#define Tk {t_k}
#define Tn {t_n}
#define SHIFT {layer['shift']}
#define DO_RELU {str(layer['is_relu']).lower()}

#include <cstdint>
__attribute__((section(".data"))) alignas(32) int8_t matB [{k_tiled.size}] = {{ {", ".join(str(int(x)) for x in k_tiled)} }};

#include "dense_i8.h"
''')

    # layer_graph.h
    num_bytes = layer['x'].size * layer['x'].itemsize
    in_port = "AIE_IN" if idx == 0 else f"layers[{idx-1}]"
    with open("model/layer_graph.h", "a") as f:
        f.write(f"layers[{idx}] = kernel::create(dense_i8);\n")
        f.write(f'source(layers[{idx}]) = "layer_{idx}.cc";\n')
        f.write(f"connect<window<{num_bytes:>5}>>({in_port}.out[0], layers[{idx}].in[0]);\n\n")


def emit_conv2d(idx, layer, params, iterations):
    """
    Emitter for AIE1 int8 conv2d kernel (m=2,n=8), macro-parameterized.

    layer['x'] : NHWC input  (XH,XW,XC) int8
    layer['k'] : weights; accepts either HWIO (KH,KW,XC,YC) or KN (KH*KW*XC, YC)
    layer['a'] : NHWC output (YH,YW,YC) int8  (reference)

    params: dict with keys:
      XH,XW,XC, KH,KW,YC, SH,SW, PAD("same"/"valid"), m(=2), n(=8), SHIFT, is_relu (bool)
    """
    XH,XW,XC   = params['XH'], params['XW'], params['XC']
    KH,KW,YC   = params['KH'], params['KW'], params['YC']
    SH,SW      = params['SH'], params['SW']
    PAD        = params['PAD']
    PH         = compute_pad(XH, KH, SH, PAD)
    PW         = compute_pad(XW, KW, SW, PAD)
    YH = (XH + 2*PH - KH)//SH + 1
    YW = (XW + 2*PW - KW)//SW + 1
    m, n       = params['m'], params['n']
    SHIFT      = params['SHIFT']
    is_relu    = str(params['is_relu']).lower()

    assert m == 2 and n == 8, "conv2d_i8: use m=2, n=8 on AIE1 int8"
    assert XC % 8 == 0 and YC % 8 == 0, "XC and YC must be multiples of 8"

    K_TOTAL = KH * KW * XC

    # ---- Pack weights to [XC//8][YC//8][KH][KW][8][8] ----
    kernel = layer['k']
    if kernel.shape == (KH, KW, XC, YC):
        W_hwio = kernel
    elif kernel.shape == (K_TOTAL, YC):
        W_hwio = kernel.reshape(KH, KW, XC, YC)
    else:
        raise ValueError(f"weights shape must be ({KH},{KW},{XC},{YC}) or ({K_TOTAL},{YC}), got {kernel.shape}")

    W6 = (W_hwio.reshape(KH,KW,XC//8,8,YC//8,8)).transpose(2,4,0,1,3,5)
    k_tiled = W6.astype(np.int8).ravel()

    # ---- I/O dumps for sim ----
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

    # ---- model.cc: instantiate conv2d ----
    with open(f"model/layer_{idx}.cc", "a") as f:
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
#define SHIFT {SHIFT}
#define DO_RELU {is_relu}

#include <cstdint>
__attribute__((section(".data"))) alignas(32) int8_t k_p [{k_tiled.size}] = {{ {", ".join(str(int(x)) for x in k_tiled)} }};

#include "conv2d_i8.h"
''')

    # ---- layer_graph wiring ----
    in_port  = "AIE_IN" if idx == 0 else f"layers[{idx-1}]"
    in_bytes = layer['x'].size * layer['x'].itemsize
    with open("model/layer_graph.h", "a") as f:
        f.write(f"layers[{idx}] = kernel::create(conv2d_i8);\n")
        f.write(f'source(layers[{idx}]) = "layer_{idx}.cc";\n')
        f.write(f"connect<window<{in_bytes:>5}>>({in_port}.out[0], layers[{idx}].in[0]);\n\n")


def emit_flatten(idx, YH, YW, YC, iterations):

    # model_fX.cc for flatten
    with open(f"model/layer_{idx}.cc", "a") as f:
        f.write(f'''
#define YH {YH}
#define YW {YW}
#define YC {YC}
#include "flatten_i8.h"
''')

    in_port = "AIE_IN" if idx == 0 else f"layers[{idx-1}]"
    in_bytes = YH*YW*YC  # bytes = count (int8)
    with open("model/layer_graph.h", "a") as f:
        f.write(f"layers[{idx}] = kernel::create(flatten_i8);\n")
        f.write(f'source(layers[{idx}]) = "layer_{idx}.cc";\n')
        f.write(f"connect<window<{in_bytes:>5}>>({in_port}.out[0], layers[{idx}].in[0]);\n\n")


# ---------------- Main ----------------

if __name__ == "__main__":
    import numpy as np, os, glob, shutil, subprocess

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

    # ----------------- Network: Conv -> (optional) Flatten -> Dense -----------------
    # Input (tiny): 4x12x8 NHWC
    XH, XW, XC = 4, 12, 8
    x0 = np.random.randint(0, 128, size=(XH,XW,XC), dtype=np.int8)

    # Conv params
    KH, KW, YC, SH, SW = 5, 7, 8, 2, 3
    SHIFT_CONV, IS_RELU = 2, False
    Wc_hwio, Wc_kn = pack_hwio_to_kn(KH, KW, XC, YC)
    y_conv = conv2d_ref(x0, Wc_hwio, stride=(SH,SW), padding="same", shift=SHIFT_CONV, relu=IS_RELU)

    YH, YW, YC = y_conv.shape
    HW = YH * YW

    # Dense params
    N1 = 8
    assert (HW % m_tile) == 0 and (YC % k_tile) == 0 and (N1 % n_tile) == 0
    Kmat1 = np.random.randint(0,128,size=(YC,N1),dtype=np.int8)

    # Flatten (reference) & Dense (reference)
    flat = y_conv.reshape(HW, YC)          # NHWC -> (HW, C)
    y_dense = (flat.astype(np.int32) @ Kmat1.astype(np.int32))
    SHIFT_DENSE = 3
    y_dense = (y_dense >> SHIFT_DENSE).astype(np.int8)  # (HW, N1)

    # ----------------- Emit layers -----------------
    layer_idx = 0

    # Conv (m=2, n=8)
    conv_params = dict(
        XH=XH, XW=XW, XC=XC, KH=KH, KW=KW, YC=YC, SH=SH, SW=SW,
        PAD="same", m=2, n=8, SHIFT=SHIFT_CONV, is_relu=IS_RELU
    )
    emit_conv2d(layer_idx, {'x': x0, 'k': Wc_kn, 'a': y_conv}, conv_params, iterations)
    layer_idx += 1

    # Optional flatten stage (explicit op boundary; bytes stay same)
    INSERT_FLATTEN = True
    if INSERT_FLATTEN:
        emit_flatten(layer_idx, YH, YW, YC, iterations)
        layer_idx += 1

    # Dense (m=2, k=8, n=8)
    dense_layer = {'x': flat, 'k': Kmat1, 'a': y_dense, 'shift': SHIFT_DENSE, 'is_relu': False}
    emit_dense(layer_idx, dense_layer, m_tile, k_tile, n_tile, iterations)
    layer_idx += 1

    # Model has conv (+optional flatten) + dense
    N_LAYERS = layer_idx
    with open("model/include.h", "w") as f:
        f.write(f'#define N_LAYERS {N_LAYERS}\n#define ITERATIONS {iterations}')

    # Final ref out: dense output tiled m=2, n=8
    tiled_last = tile_matrix(y_dense, m_tile, n_tile)
    np.savetxt("data/out_ref.txt",
               np.tile(tiled_last, (iterations,1)).reshape(-1,16),
               fmt="%s", delimiter=" ")

    # Close graph
    out_bytes = y_dense.size * y_dense.itemsize
    with open("model/layer_graph.h", "a") as f:
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
