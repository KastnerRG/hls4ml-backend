#!/usr/bin/env python3
"""Verify aiesimulator output against numpy reference; print latency report."""
import glob, json, os, re, sys
import numpy as np

def read_plio(path, shape):
    with open(path) as f:
        lines = [l for l in f if not l.startswith("T")]
    return np.array([int(v) for l in lines for v in l.split()],
                    dtype=np.int32).reshape(shape).astype(np.uint8)

def pl_latency_cycles():
    """Parse top-level HLS synthesis reports; return total latency cycles, or 0."""
    total = 0
    for rpt in sorted(glob.glob("pl_group*/hls/syn/report/pl_group*_csynth.rpt")):
        if not re.fullmatch(r'pl_group\d+_csynth\.rpt', os.path.basename(rpt)):
            continue
        with open(rpt) as f:
            text = f.read()
        # Summary row: | lat_min | lat_max | ... us | ... us | ii_min | ii_max | Type |
        m = re.search(r'\|\s*\d+\|\s*(\d+)\|[^|]+\|[^|]+\|\s*\d+\|\s*\d+\|', text)
        if m:
            total += int(m.group(1))
    return total

cfg = json.load(open("data/config.json")) if os.path.exists("data/config.json") else {}
n_aie = cfg.get("n_aie_groups", 0)
if not n_aie:
    sys.exit("No AIE groups in config — run 'make gen' first.")

ok = True
total_cycles = 0
for gi in range(n_aie):
    ref_path = f"data/g{gi}_ref.npy"
    if not os.path.exists(ref_path):
        sys.exit(f"Missing {ref_path} — run 'make gen' first.")
    ref = np.load(ref_path)
    out_path = f"aiesimulator_output/data/g{gi}_ofm.txt"
    if not os.path.exists(out_path):
        print(f"  ✗ group {gi}: sim output missing"); ok = False; continue
    out = read_plio(out_path, ref.shape)
    if np.array_equal(out, ref):
        print(f"  ✓ group {gi}: {ref.shape}")
    else:
        diff = np.abs(out.astype(np.int32) - ref.astype(np.int32))
        print(f"  ✗ group {gi}: max_diff={diff.max():.0f}  mean={diff.mean():.3f}")
        print(f"    ref[0]={ref[0]}  sim[0]={out[0]}")
        ok = False
    lat = f"aiesimulator_output/data/g{gi}_latency.json"
    if os.path.exists(lat):
        total_cycles += json.load(open(lat))["cycles"]

B         = cfg.get("batch", "?")
AIE_GHZ, PL_MHZ = 1.25, 312.5

pl_cycles = pl_latency_cycles()
pl_label  = "HLS report" if pl_cycles else "est."
if not pl_cycles:
    pl_cycles = cfg.get("pl_cycles_est", 0)

print()
if total_cycles:
    aie_ns = total_cycles / AIE_GHZ
    print(f"AIE : {total_cycles} cycles  ({aie_ns:.1f} ns @ {AIE_GHZ} GHz)  [aiesimulator]")
if pl_cycles:
    pl_ns = pl_cycles / (PL_MHZ * 1e6) * 1e9
    print(f"PL  : {pl_cycles} cycles  ({pl_ns:.1f} ns @ {PL_MHZ} MHz)  [{pl_label}]")
    if total_cycles:
        total_ns = aie_ns + pl_ns
        print(f"Total (analytical): {total_ns:.1f} ns  →  {B / (total_ns * 1e-9) / 1e6:.2f} M samples/s")

sys.exit(0 if ok else 1)
