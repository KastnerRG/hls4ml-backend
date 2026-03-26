#!/usr/bin/env python3

import numpy as np
import os, subprocess, json, textwrap

# ── Knobs ──────────────────────────────────────────────────────────────────────
BATCH        = 8
FEAT         = 64     # input and output features per layer (same for all 4 layers)
SHIFT        = 0      # right-shift after accumulation (0 = no rounding, safe numpy match)
N_ITER       = 1

# AIE knobs (layers 3–4)
CAS_LENGTH   = 1      # K-reduction cascade depth (1 = single tile per output chain)
CAS_NUM      = 1      # output feature partitions  (1 = all features in one tile)
TILE_M       = 4      # mmul M dimension
TILE_K       = 8      # mmul K dimension
TILE_N       = 8      # mmul N dimension
COL_L3       = 1      # AIE tile column for layer 3
ROW_L3       = 0
COL_L4       = 3      # AIE tile column for layer 4
ROW_L4       = 0

# HLS knobs (layers 1–2, for synthesis reference)
REUSE_FACTOR = 1

# Derived (must satisfy aie4ml static asserts)
IN_FEAT_SLICE  = FEAT // CAS_LENGTH
OUT_FEAT_SLICE = FEAT // CAS_NUM
assert IN_FEAT_SLICE  % (2 * TILE_K) == 0, "IN_FEAT_SLICE must be divisible by 2*TILE_K"
assert OUT_FEAT_SLICE % (2 * TILE_N) == 0, "OUT_FEAT_SLICE must be divisible by 2*TILE_N"
assert BATCH          % (2 * TILE_M) == 0, "BATCH must be divisible by 2*TILE_M"
# ──────────────────────────────────────────────────────────────────────────────


def dense_relu_int8(x, W, b, shift):
    """PL layer: int8 input → int8 output, relu, saturate to [0, 127]."""
    acc = x.astype(np.int32) @ W.astype(np.int32) + b.astype(np.int32)
    return np.clip(np.maximum(0, acc >> shift), 0, 127).astype(np.int8)


def dense_relu_uint8(x, W, b, shift):
    """AIE layer: int8/uint8 input → uint8 output, relu, saturate to [0, 255]."""
    acc = x.astype(np.int32) @ W.astype(np.int32) + b.astype(np.int32)
    return np.clip(np.maximum(0, acc >> shift), 0, 255).astype(np.uint8)


def pack_weights(W, K, N):
    """Pack (IN_FEAT, OUT_FEAT) int8 weights into (K_tile, N_tile, K, N) order."""
    KF, NF = W.shape
    return W.reshape(KF // K, K, NF // N, N).transpose(0, 2, 1, 3).flatten()


def write_weight_header(path, name, W_packed, cas_num, cas_length, in_slice, out_slice):
    W_split = W_packed.reshape(cas_num, cas_length, in_slice * out_slice)
    with open(path, "w") as f:
        f.write(f"int8_t {name}[{cas_num}][{cas_length}][{in_slice * out_slice}] = {{\n")
        for ch in range(cas_num):
            f.write("  {\n")
            for col in range(cas_length):
                vals = ", ".join(str(int(v)) for v in W_split[ch][col])
                f.write(f"    {{{vals}}}\n")
            f.write("  },\n")
        f.write("};\n")


def write_bias_header(path, name, bias, cas_num, out_slice):
    b_split = bias.reshape(cas_num, out_slice)
    with open(path, "w") as f:
        f.write(f"int32_t {name}[{cas_num}][{out_slice}] = {{\n")
        for ch in range(cas_num):
            vals = ", ".join(str(int(v)) for v in b_split[ch])
            f.write(f"  {{{vals}}},\n")
        f.write("};\n")


def write_parameters_h(path):
    def cfg(name, data_t, col, row):
        return textwrap.dedent(f"""\
            struct {name} {{
              using data_t        = {data_t};
              using weight_t      = int8_t;
              using result_t      = uint8_t;
              using bias_t        = int32_t;
              using acc_scalar_t  = acc32;
              static constexpr int IN_FEAT  = {FEAT};
              static constexpr int OUT_FEAT = {FEAT};
              static constexpr int CAS_LENGTH = {CAS_LENGTH};
              static constexpr int CAS_NUM    = {CAS_NUM};
              static constexpr bool USE_BIAS        = true;
              static constexpr bool USE_RELU        = true;
              static constexpr bool TRANSPOSE_INPUT = false;
              static constexpr int SHIFT = {SHIFT};
              static constexpr int M = {TILE_M}, K = {TILE_K}, N = {TILE_N};
              static constexpr int col_placement = {col}, row_placement = {row};
              static constexpr int padded_independent_extent = {BATCH};
              static constexpr int padded_IN_FEAT  = {FEAT};
              static constexpr int padded_OUT_FEAT = {FEAT};
              static constexpr int IN_FEAT_SLICE   = {IN_FEAT_SLICE};
              static constexpr int OUT_FEAT_SLICE  = {OUT_FEAT_SLICE};
              static constexpr int RAW_IN_FEAT_SLICE  = {IN_FEAT_SLICE};
              static constexpr int RAW_OUT_FEAT_SLICE = {OUT_FEAT_SLICE};
            #if __cplusplus >= 202002L
              static constexpr auto ROUNDING   = aie::rounding_mode::conv_even;
              static constexpr auto SATURATION = aie::saturation_mode::saturate;
            #endif
              static constexpr const char* ROUNDING_TOKEN   = "conv_even";
              static constexpr const char* SATURATION_TOKEN = "saturate";
            }};
            """)
    with open(path, "w") as f:
        f.write("#pragma once\n")
        f.write("#include <adf.h>\n")
        f.write("#include <aie_api/aie.hpp>\n")
        f.write("#include <cstdint>\n\n")
        f.write(f"#define N_ITER {N_ITER}\n\n")
        f.write(cfg("L3Cfg", "int8_t",  COL_L3, ROW_L3))
        f.write("\n")
        f.write(cfg("L4Cfg", "uint8_t", COL_L4, ROW_L4))


def write_plio(path, x):
    """Write (BATCH, FEAT) array to PLIO txt.
    PLIO order is batch-major (matches aie4ml convention).
    16 int8 values per 128-bit PLIO word."""
    flat = x.flatten().astype(np.int32)   # (BATCH, FEAT) → batch-major
    np.savetxt(path, flat.reshape(-1, 16), fmt="%d")


def read_plio(path):
    """Read PLIO txt back to (BATCH, FEAT) uint8 array."""
    with open(path) as f:
        lines = [l for l in f if not l.startswith("T")]
    flat = np.array([int(v) for l in lines for v in l.split()], dtype=np.int32)
    return flat.reshape(BATCH, FEAT).astype(np.uint8)   # batch-major → (BATCH, FEAT)


# ── Main ───────────────────────────────────────────────────────────────────────

np.random.seed(42)
os.makedirs("data",        exist_ok=True)
os.makedirs("aie/weights", exist_ok=True)

# Random int8 weights (small range so no saturation with SHIFT=0 for small inputs)
W1 = np.random.randint(-3, 4, (FEAT, FEAT), dtype=np.int8)
W2 = np.random.randint(-3, 4, (FEAT, FEAT), dtype=np.int8)
W3 = np.random.randint(-3, 4, (FEAT, FEAT), dtype=np.int8)
W4 = np.random.randint(-3, 4, (FEAT, FEAT), dtype=np.int8)
b1 = np.zeros(FEAT, dtype=np.int32)
b2 = np.zeros(FEAT, dtype=np.int32)
b3 = np.zeros(FEAT, dtype=np.int32)
b4 = np.zeros(FEAT, dtype=np.int32)

x0 = np.random.randint(0, 4, (BATCH, FEAT), dtype=np.int8)  # small inputs

# ── PL layers (numpy reference, int8 output) ──
print("[PL] Layer 1: dense 64→64 relu (HLS, numpy ref)")
y1 = dense_relu_int8(x0, W1, b1, SHIFT)

print("[PL] Layer 2: dense 64→64 relu (HLS, numpy ref)")
y2 = dense_relu_int8(y1, W2, b2, SHIFT)

# ── AIE reference (numpy, uint8 output) ──
y3_ref = dense_relu_uint8(y2, W3, b3, SHIFT)
y4_ref = dense_relu_uint8(y3_ref, W4, b4, SHIFT)

# ── Write generated files ──
write_plio("data/ifm.txt", y2)
write_parameters_h("aie/parameters.h")

write_weight_header("aie/weights/weights_l3.h", "weights_l3",
    pack_weights(W3, TILE_K, TILE_N), CAS_NUM, CAS_LENGTH, IN_FEAT_SLICE, OUT_FEAT_SLICE)
write_bias_header("aie/weights/bias_l3.h", "bias_l3", b3, CAS_NUM, OUT_FEAT_SLICE)

write_weight_header("aie/weights/weights_l4.h", "weights_l4",
    pack_weights(W4, TILE_K, TILE_N), CAS_NUM, CAS_LENGTH, IN_FEAT_SLICE, OUT_FEAT_SLICE)
write_bias_header("aie/weights/bias_l4.h", "bias_l4", b4, CAS_NUM, OUT_FEAT_SLICE)

# ── Compile + simulate AIE ──
print("\n[AIE] Compiling and simulating (make sim)...")
_env = os.environ.copy()
# system readelf needs libdebuginfod.so.1; point it to the conda-installed copy
import sysconfig
_conda_lib = sysconfig.get_path("stdlib").rsplit("/lib/", 1)[0] + "/lib"
_env["LD_LIBRARY_PATH"] = _conda_lib + (":" + _env["LD_LIBRARY_PATH"] if _env.get("LD_LIBRARY_PATH") else "")
subprocess.run(["make", "sim"], check=True, env=_env)

# ── Verify AIE output ──
aie_out_path = "aiesimulator_output/data/out.txt"
assert os.path.exists(aie_out_path), f"AIE output not found: {aie_out_path}"
y4_sim = read_plio(aie_out_path)

if np.array_equal(y4_sim, y4_ref):
    print(f"\n Success: AIE output matches reference {y4_sim.shape}")
else:
    diff = np.abs(y4_sim.astype(np.int32) - y4_ref.astype(np.int32))
    print(f"\n Mismatch: max diff = {diff.max()}, mean diff = {diff.mean():.3f}")
    print(f"  sim[0]: {y4_sim[0]}")
    print(f"  ref[0]: {y4_ref[0]}")

# ── Latency & throughput report ──
print("\n=== Latency Report ===")

latency_path = "aiesimulator_output/data/latency.json"
aie_cycles = None
if os.path.exists(latency_path):
    with open(latency_path) as f:
        aie_cycles = json.load(f)["cycles"]

AIE_FREQ_GHZ = 1.25
PL_FREQ_MHZ  = 312.5
pl_cycles_est = 2 * BATCH * FEAT // REUSE_FACTOR  # pipeline: BATCH iterations of n_in-cycle loop

if aie_cycles is not None:
    aie_ns = aie_cycles / AIE_FREQ_GHZ
    print(f"AIE latency (layers 3–4):  {aie_cycles} cycles  ({aie_ns:.1f} ns @ {AIE_FREQ_GHZ} GHz)")

pl_ns = pl_cycles_est / (PL_FREQ_MHZ * 1e6) * 1e9
print(f"PL  latency (layers 1–2):  ~{pl_cycles_est} cycles  ({pl_ns:.1f} ns @ {PL_FREQ_MHZ} MHz, estimated)")

if aie_cycles is not None:
    total_ns = aie_ns + pl_ns
    throughput_m = BATCH / (total_ns * 1e-9) / 1e6
    print(f"Total latency:             {total_ns:.1f} ns")
    print(f"Throughput:                {throughput_m:.2f} M samples/sec  ({BATCH} samples / {total_ns:.1f} ns)")
