#!/usr/bin/env python3
"""
exp_aie_pl_model_sweep.py
Sweep PL-AIE crossings for a 16-layer int8 dense NN.
  - 16 layers total: 8 PL + 8 AIE, each 128→128, batch=8, ReLU
  - First and last layers always PL
  - x-axis: number of PL-AIE crossings (2, 4, 6, 8, 10, 12, 14)
  - For each config: make clean → make sim → make run_emu
  - Collects AIE + PL analytical latency (cycles), saves CSV, plots graph

Usage:
    cd /path/to/hls4ml-backend
    python exp_aie_pl_model_sweep.py
"""
import os, re, json, csv, subprocess, sys, shutil, time

# ── Config ────────────────────────────────────────────────────────────────────
AIE_DIR   = os.path.join(os.path.dirname(os.path.abspath(__file__)), "aie_pl")
RUNS_DIR  = os.path.join(AIE_DIR, "runs")
GEN_PY    = os.path.join(AIE_DIR, "gen.py")

BATCH        = 8
FEAT         = 128
N_LAYERS     = 16     # total layers
N_PL         = 8      # PL layers (fixed)
N_AIE        = 8      # AIE layers (fixed)
REUSE_FACTOR = 128    # HLS reuse factor (1=max DSP, 128=1/128 DSPs; 128×128/RF DSPs per group)
SHIFT        = 7      # right-shift for fixed-point accumulator

CROSSINGS = [2, 4, 6, 8, 10, 12, 14]

AIE_GHZ   = 1.25
PL_MHZ    = 312.5

# ── Helpers ───────────────────────────────────────────────────────────────────

def distribute(total, groups):
    """Distribute `total` items into `groups` groups as evenly as possible (front-heavy)."""
    base, rem = divmod(total, groups)
    return [base + (1 if i < rem else 0) for i in range(groups)]


def make_layers(n_crossings):
    """
    Build LAYERS list for a given crossing count.
    n_crossings must be even (first + last layer always PL).
    n_crossings/2 = number of AIE segments.
    """
    k = n_crossings // 2          # number of AIE segments
    aie_sizes = distribute(N_AIE, k)       # layers per AIE segment
    pl_sizes  = distribute(N_PL,  k + 1)  # layers per PL segment (k+1 PL segments)
    layers = []
    for i in range(k + 1):
        for _ in range(pl_sizes[i]):
            layers.append(("pl",  FEAT, FEAT))
        if i < k:
            for _ in range(aie_sizes[i]):
                layers.append(("aie", FEAT, FEAT))
    assert len(layers) == N_LAYERS, f"Layer count mismatch: {len(layers)}"
    assert layers[0][0] == "pl" and layers[-1][0] == "pl"
    return layers


def patch_gen_layers(layers):
    """Overwrite the LAYERS = [...] block in gen.py with the new layer list."""
    with open(GEN_PY) as f:
        text = f.read()
    inner = "\n".join(f'    ("{t}", {ni}, {no}),' for t, ni, no in layers)
    new_block = f"LAYERS = [\n{inner}\n]"
    text = re.sub(r"LAYERS\s*=\s*\[.*?\]", new_block, text, flags=re.DOTALL)
    with open(GEN_PY, "w") as f:
        f.write(text)


MAKE_ENV_OVERRIDES = {
    "BATCH":        str(BATCH),
    "REUSE_FACTOR": str(REUSE_FACTOR),
    "SHIFT":        str(SHIFT),
}


def run_make(target, run_dir, timeout=14400):
    """Run make <target> in AIE_DIR with sweep knobs; save log to run_dir/<target>.log."""
    log_path = os.path.join(run_dir, f"{target.replace(' ', '_')}.log")
    env = os.environ.copy()
    env.update(MAKE_ENV_OVERRIDES)
    t0 = time.time()
    proc = subprocess.run(
        ["make", target],
        cwd=AIE_DIR,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout,
        text=True,
    )
    elapsed = time.time() - t0
    with open(log_path, "w") as f:
        f.write(proc.stdout)
    return proc.returncode, proc.stdout, elapsed


def read_check_py(run_dir):
    """
    Run check.py in AIE_DIR, capture output.
    Returns dict with aie_cycles, pl_cycles, total_ns, passed.
    """
    proc = subprocess.run(
        [sys.executable, os.path.join(AIE_DIR, "check.py")],
        cwd=AIE_DIR,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    out = proc.stdout
    with open(os.path.join(run_dir, "check.log"), "w") as f:
        f.write(out)

    aie_cyc = pl_cyc = total_ns = 0
    m = re.search(r"AIE\s*:\s*(\d+)\s*cycles", out)
    if m: aie_cyc = int(m.group(1))
    m = re.search(r"PL\s*:\s*(\d+)\s*cycles", out)
    if m: pl_cyc = int(m.group(1))
    m = re.search(r"Total.*?:\s*([\d.]+)\s*ns", out)
    if m: total_ns = float(m.group(1))

    # If only AIE is available (no PL HLS report yet), compute total from estimate
    if aie_cyc and not total_ns:
        total_ns = aie_cyc / AIE_GHZ  # minimum (no PL latency)

    passed = proc.returncode == 0
    return {"aie_cycles": aie_cyc, "pl_cycles": pl_cyc,
            "total_ns": total_ns, "check_passed": passed, "check_output": out}


def check_hw_emu_pass(run_dir):
    """Check hw_emu output log for PASS/FAIL."""
    emu_log = os.path.join(run_dir, "run_emu.log")
    if not os.path.exists(emu_log):
        return None
    with open(emu_log) as f:
        content = f.read()
    if "PASS" in content:
        return True
    if "FAIL" in content:
        return False
    return None


# ── Main sweep ────────────────────────────────────────────────────────────────

def main():
    os.makedirs(RUNS_DIR, exist_ok=True)
    results = []
    original_layers = None

    # Save original gen.py LAYERS so we can restore at end
    with open(GEN_PY) as f:
        gen_original = f.read()

    print(f"{'='*65}")
    print(f"  PL-AIE Crossing Sweep: {N_LAYERS}-layer {FEAT}×{FEAT} dense, batch={BATCH}")
    print(f"  Crossings: {CROSSINGS}")
    print(f"  Results: {RUNS_DIR}")
    print(f"{'='*65}")

    for n_cross in CROSSINGS:
        run_dir = os.path.join(RUNS_DIR, f"crossings_{n_cross:02d}")
        os.makedirs(run_dir, exist_ok=True)

        layers = make_layers(n_cross)
        k = n_cross // 2

        print(f"\n[crossings={n_cross}]  {k} AIE group(s), {k+1} PL group(s)")
        # Print segment summary
        seg_summary = []
        cur = layers[0][0]; cnt = 1
        for t, _, _ in layers[1:]:
            if t == cur: cnt += 1
            else: seg_summary.append(f"{cur.upper()}×{cnt}"); cur = t; cnt = 1
        seg_summary.append(f"{cur.upper()}×{cnt}")
        print(f"  Sequence: {' → '.join(seg_summary)}")

        # Save layer config
        with open(os.path.join(run_dir, "layers.json"), "w") as f:
            json.dump({"n_crossings": n_cross, "layers": layers}, f, indent=2)

        # Patch gen.py
        patch_gen_layers(layers)

        # ── make clean ────────────────────────────────────────────────────────
        print("  make clean ...", flush=True)
        rc, _, _ = run_make("clean", run_dir, timeout=120)

        # ── make sim ──────────────────────────────────────────────────────────
        print("  make sim  ...", flush=True)
        rc_sim, out_sim, t_sim = run_make("sim", run_dir, timeout=7200)
        print(f"  make sim  done ({t_sim/60:.1f} min, rc={rc_sim})")
        if rc_sim != 0:
            print("  ERROR: make sim failed — skipping this crossing")
            tail = out_sim[-1000:] if len(out_sim) > 1000 else out_sim
            print("  Last output:\n" + tail)
            results.append({"crossings": n_cross, "status": "sim_fail",
                            "aie_cycles": 0, "pl_cycles": 0, "total_ns": 0,
                            "sim_time_min": t_sim/60, "emu_time_min": 0,
                            "hw_emu_pass": None})
            continue

        # ── check.py after sim (AIE latency; PL estimated) ───────────────────
        chk_sim = read_check_py(run_dir)
        print(f"  AIE: {chk_sim['aie_cycles']} cycles ({chk_sim['aie_cycles']/AIE_GHZ:.1f} ns)")

        # ── make run_emu ──────────────────────────────────────────────────────
        print("  make run_emu ...", flush=True)
        rc_emu, out_emu, t_emu = run_make("run_emu", run_dir, timeout=14400)
        print(f"  make run_emu done ({t_emu/60:.1f} min, rc={rc_emu})")

        # ── check.py after run_emu (AIE + PL HLS report) ─────────────────────
        shutil.copy(os.path.join(run_dir, "check.log"),
                    os.path.join(run_dir, "check_post_sim.log"))
        chk_emu = read_check_py(run_dir)
        shutil.copy(os.path.join(run_dir, "check.log"),
                    os.path.join(run_dir, "check_post_emu.log"))

        aie_cyc  = chk_emu["aie_cycles"] or chk_sim["aie_cycles"]
        pl_cyc   = chk_emu["pl_cycles"]
        total_ns = chk_emu["total_ns"]
        hw_pass  = check_hw_emu_pass(run_dir)

        if not total_ns and aie_cyc:
            aie_ns   = aie_cyc / AIE_GHZ
            pl_ns    = pl_cyc / (PL_MHZ * 1e6) * 1e9 if pl_cyc else 0
            total_ns = aie_ns + pl_ns

        status = "pass" if (chk_emu["check_passed"] and hw_pass) else \
                 ("emu_pass" if chk_emu["check_passed"] else "fail")

        print(f"  AIE: {aie_cyc} cyc ({aie_cyc/AIE_GHZ:.1f} ns)  "
              f"PL: {pl_cyc} cyc ({pl_cyc/(PL_MHZ*1e6)*1e9:.1f} ns)  "
              f"Total: {total_ns:.1f} ns  hw_emu={hw_pass}")

        row = {
            "crossings":     n_cross,
            "status":        status,
            "aie_cycles":    aie_cyc,
            "pl_cycles":     pl_cyc,
            "total_ns":      total_ns,
            "sim_time_min":  round(t_sim / 60, 1),
            "emu_time_min":  round(t_emu / 60, 1),
            "hw_emu_pass":   hw_pass,
        }
        results.append(row)
        with open(os.path.join(run_dir, "results.json"), "w") as f:
            json.dump(row, f, indent=2)

    # ── Restore original gen.py ────────────────────────────────────────────────
    with open(GEN_PY, "w") as f:
        f.write(gen_original)
    print("\n[gen.py restored to original LAYERS]")

    # ── Save CSV ───────────────────────────────────────────────────────────────
    csv_path = os.path.join(RUNS_DIR, "sweep_results.csv")
    if results:
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=results[0].keys())
            w.writeheader()
            w.writerows(results)
        print(f"CSV saved: {csv_path}")

    # ── Plot ───────────────────────────────────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        valid = [r for r in results if r["total_ns"] > 0]
        if valid:
            xs   = [r["crossings"] for r in valid]
            ys   = [r["total_ns"]  for r in valid]
            aie_ns_list = [r["aie_cycles"] / AIE_GHZ for r in valid]
            pl_ns_list  = [r["pl_cycles"] / (PL_MHZ * 1e6) * 1e9 for r in valid]

            fig, ax = plt.subplots(figsize=(9, 5))
            ax.plot(xs, ys,          "o-", color="C0", linewidth=2, markersize=8, label="Total")
            ax.plot(xs, aie_ns_list, "s--", color="C1", linewidth=1.5, markersize=6, label="AIE only")
            ax.plot(xs, pl_ns_list,  "^--", color="C2", linewidth=1.5, markersize=6, label="PL only")
            ax.set_xlabel("Number of PL↔AIE Crossings", fontsize=12)
            ax.set_ylabel("Latency (ns)", fontsize=12)
            ax.set_title(f"16-layer Dense NN: Latency vs PL-AIE Crossings\n"
                         f"(batch={BATCH}, {FEAT}×{FEAT} int8, VEK280)", fontsize=12)
            ax.set_xticks(CROSSINGS)
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plot_path = os.path.join(RUNS_DIR, "latency_vs_crossings.png")
            plt.savefig(plot_path, dpi=150)
            plt.close()
            print(f"Plot saved: {plot_path}")
    except ImportError:
        print("matplotlib not available — skipping plot")

    # ── Print summary table ────────────────────────────────────────────────────
    print(f"\n{'─'*65}")
    print(f"{'Crossings':>10}  {'AIE cyc':>9}  {'PL cyc':>8}  {'Total ns':>10}  Status")
    print(f"{'─'*65}")
    for r in results:
        print(f"  {r['crossings']:>8}  {r['aie_cycles']:>9}  {r['pl_cycles']:>8}  "
              f"{r['total_ns']:>10.1f}  {r['status']}")
    print(f"{'─'*65}")


if __name__ == "__main__":
    main()
