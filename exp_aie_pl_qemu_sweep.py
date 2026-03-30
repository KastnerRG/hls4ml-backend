#!/usr/bin/env python3
"""
exp_aie_pl_qemu_sweep.py
hw_emu (QEMU) sweep of PL-AIE crossings for a 16-layer int8 dense NN.

For each crossing config:
  1. make clean + make hw_emu  (compile kernels, link, package — ~30–60 min)
  2. make run_emu × QEMU_RUNS  (boot QEMU, run host.exe, capture timing — ~10–20 min each)
  3. Parse "End-to-end (hw_emu): X ns" from QEMU stdout

Results → runs/run_qemu/sweep_qemu.csv

Design notes:
  Timing resolution vs wall-clock tradeoff:
    QUANTUM_NS controls the xsim↔QEMU sync quantum. Values < 10000 ns cause
    race conditions between the APU and PMC QEMU processes during zynqmp clock
    probe, hard-deadlocking the boot. QUANTUM_NS=100000 (100 μs) is the minimum
    stable value (the original developer-tested default).

  Signal amplification:
    Per-inference latency is 3500–5200 ns; adjacent-crossing differences are
    150–350 ns. At 1 μs quantum this is <1 quantum — unresolvable with one run.
    N_ITER=100: host.cpp loops 100 full pipeline runs per measurement, prints
    per-iteration average. The timed window is 350–520 μs (350–520 quanta at
    1 μs), giving 100–170 quanta difference between crossing=2 and crossing=14.

  SW overhead:
    graph.update() (weight loading) is outside the timed region.
    A warmup iteration runs before the timed loop to settle pipeline state.
    XRT kernel launch overhead is constant across crossing configs.

Usage:
    cd /path/to/hls4ml-backend
    python exp_aie_pl_qemu_sweep.py
"""
import glob, os, re, json, csv, shutil, subprocess, sys, time, statistics

# ── Config ────────────────────────────────────────────────────────────────────
AIE_DIR   = os.path.join(os.path.dirname(os.path.abspath(__file__)), "aie_pl")
RUNS_DIR  = os.path.join(AIE_DIR, "runs", "run_qemu")
GEN_PY    = os.path.join(AIE_DIR, "gen.py")

BATCH        = 8
FEAT         = 48
N_LAYERS     = 16
N_PL         = 8
N_AIE        = 8
REUSE_FACTOR = 1
SHIFT        = 0

CROSSINGS    = [2, 4, 6, 8, 10, 12, 14]
QEMU_RUNS    = 3      # QEMU boots per crossing config; report median
QUANTUM_NS   = 100000 # simulation time quantum (ns): 100 μs — stable boot, xsim sync
N_ITER       = 100    # pipeline iterations per timed measurement (signal amplification)
QEMU_TIMEOUT = 1800   # seconds per make run_emu invocation (30 min)


# ── Layer assignment (identical to exp_aie_pl_model_sweep.py) ─────────────────

def distribute(total, groups):
    base, rem = divmod(total, groups)
    return [base + (1 if i < rem else 0) for i in range(groups)]


def make_layers(n_crossings):
    assert n_crossings % 2 == 0
    k = n_crossings // 2
    pl_sizes  = distribute(N_PL,  k + 1)
    aie_sizes = distribute(N_AIE, k)

    layers = []
    for i in range(k + 1):
        for _ in range(pl_sizes[i]):
            layers.append(("pl",  FEAT, FEAT))
        if i < k:
            for _ in range(aie_sizes[i]):
                layers.append(("aie", FEAT, FEAT))

    assert len(layers) == N_LAYERS
    assert layers[0][0] == "pl" and layers[-1][0] == "pl"
    return layers


def seg_summary(layers):
    segs = []
    cur = layers[0][0]; cnt = 1
    for t, _, _ in layers[1:]:
        if t == cur:
            cnt += 1
        else:
            segs.append(f"{cur.upper()}×{cnt}"); cur = t; cnt = 1
    segs.append(f"{cur.upper()}×{cnt}")
    return " → ".join(segs)


def patch_gen_layers(layers):
    with open(GEN_PY) as f:
        text = f.read()
    inner = "\n".join(f'    ("{t}", {ni}, {no}),' for t, ni, no in layers)
    new_block = f"LAYERS = [\n{inner}\n]"
    text = re.sub(r"LAYERS\s*=\s*\[.*?\]", new_block, text, flags=re.DOTALL)
    with open(GEN_PY, "w") as f:
        f.write(text)


# ── Make helpers ──────────────────────────────────────────────────────────────

MAKE_ENV = {
    "BATCH":        str(BATCH),
    "REUSE_FACTOR": str(REUSE_FACTOR),
    "SHIFT":        str(SHIFT),
    "QUANTUM_NS":   str(QUANTUM_NS),
    "N_ITER":       str(N_ITER),
}


def run_make(target, log_path, timeout=None):
    env = os.environ.copy()
    env.update(MAKE_ENV)
    t0 = time.time()
    try:
        proc = subprocess.run(
            ["make", target],
            cwd=AIE_DIR, env=env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, timeout=timeout,
        )
        rc = proc.returncode
        out = proc.stdout
    except subprocess.TimeoutExpired as e:
        out = (e.stdout or b"").decode(errors="replace")
        rc = -1
        out += f"\n[TIMEOUT after {timeout}s]\n"
    elapsed = time.time() - t0
    with open(log_path, "w") as f:
        f.write(out)
    return rc, out, elapsed


# ── QEMU output parsing ───────────────────────────────────────────────────────

def parse_qemu_output(text):
    """
    Extract (e2e_ns, passed) from make run_emu stdout.

    host.exe prints:
        End-to-end (hw_emu): 12345.6 ns
        PASS
    """
    m = re.search(r"End-to-end \(hw_emu\):\s*([\d.eE+\-]+)\s*ns", text)
    e2e_ns = float(m.group(1)) if m else None
    passed = "PASS" in text and "FAIL" not in text
    return e2e_ns, passed


# ── Artifact archiving ────────────────────────────────────────────────────────

def archive_build(run_dir):
    """Save hw_emu build artifacts needed to re-run QEMU later."""
    # sw/ contains launch_hw_emu.sh, qemu_args.txt, sd_card image, host.exe
    sw_src = os.path.join(AIE_DIR, "sw")
    sw_dst = os.path.join(run_dir, "sw")
    if os.path.isdir(sw_src):
        if os.path.exists(sw_dst):
            shutil.rmtree(sw_dst)
        shutil.copytree(sw_src, sw_dst)

    # Package and link logs
    for name in ("package.log", "link.log", "host.log"):
        src = os.path.join(AIE_DIR, name)
        if os.path.exists(src):
            shutil.copy2(src, run_dir)


# ── Main sweep ────────────────────────────────────────────────────────────────

def main():
    os.makedirs(RUNS_DIR, exist_ok=True)
    results = []

    with open(GEN_PY) as f:
        gen_original = f.read()

    print(f"{'='*70}")
    print(f"  PL-AIE Crossing Sweep — hw_emu (QEMU)")
    print(f"  Model: {N_LAYERS} layers × (batch={BATCH}, {FEAT}→{FEAT}) int8, RF={REUSE_FACTOR}")
    print(f"  Crossings: {CROSSINGS}")
    print(f"  QUANTUM_NS={QUANTUM_NS}  QEMU_RUNS={QEMU_RUNS}")
    print(f"  Results → {RUNS_DIR}")
    print(f"{'='*70}")

    for n_cross in CROSSINGS:
        run_dir = os.path.join(RUNS_DIR, f"crossing_{n_cross}")
        os.makedirs(run_dir, exist_ok=True)

        layers  = make_layers(n_cross)
        summary = seg_summary(layers)
        k       = n_cross // 2

        print(f"\n[crossings={n_cross}]  {summary}")

        with open(os.path.join(run_dir, "layers.json"), "w") as f:
            json.dump({"n_crossings": n_cross, "sequence": summary,
                       "n_pl": N_PL, "n_aie": N_AIE,
                       "batch": BATCH, "feat": FEAT, "layers": layers}, f, indent=2)

        patch_gen_layers(layers)

        # ── Phase 1: compile ──────────────────────────────────────────────────
        print("  make clean ...", flush=True)
        run_make("clean", os.path.join(run_dir, "clean.log"))

        print("  make hw_emu ...", flush=True)
        rc_hw, out_hw, t_hw = run_make("hw_emu", os.path.join(run_dir, "hw_emu.log"))
        print(f"  make hw_emu done  ({t_hw/60:.1f} min, rc={rc_hw})")

        if rc_hw != 0:
            print("  ERROR: make hw_emu failed")
            print("  Last 800 chars:\n" + out_hw[-800:])
            results.append({
                "crossings": n_cross, "sequence": summary, "status": "FAIL_BUILD",
                "e2e_ns_median": "", "e2e_ns_min": "", "e2e_ns_all": "",
                "build_time_min": round(t_hw / 60, 1),
            })
            continue

        archive_build(run_dir)

        # ── Phase 2: run QEMU N times ─────────────────────────────────────────
        e2e_samples = []
        all_passed  = True

        for run_i in range(QEMU_RUNS):
            print(f"  make run_emu  [{run_i+1}/{QEMU_RUNS}] ...", flush=True)
            rc_run, out_run, t_run = run_make(
                "run_emu",
                os.path.join(run_dir, f"run_emu_{run_i}.log"),
                timeout=QEMU_TIMEOUT,
            )
            # Always kill orphaned QEMU/xsim processes before the next boot.
            # launch_hw_emu.sh spawns background processes; killing `make` alone
            # leaves them running and they corrupt subsequent QEMU boots.
            run_make("killemu", os.path.join(run_dir, f"killemu_{run_i}.log"))

            e2e_ns, passed = parse_qemu_output(out_run)
            status = "ok" if (rc_run == 0 and passed) else "FAIL"
            print(f"  run {run_i+1}: {e2e_ns} ns  {status}  ({t_run/60:.1f} min)")

            if e2e_ns is not None:
                e2e_samples.append(e2e_ns)
            if not passed:
                all_passed = False

        if not e2e_samples:
            print("  ERROR: no QEMU timing samples captured")
            results.append({
                "crossings": n_cross, "sequence": summary, "status": "FAIL_RUN",
                "e2e_ns_median": "", "e2e_ns_min": "", "e2e_ns_all": "",
                "build_time_min": round(t_hw / 60, 1),
            })
            continue

        e2e_median = round(statistics.median(e2e_samples), 1)
        e2e_min    = round(min(e2e_samples), 1)
        e2e_str    = " | ".join(f"{v:.1f}" for v in e2e_samples)
        status     = "pass" if all_passed else "mismatch"

        print(f"  End-to-end: median={e2e_median} ns  min={e2e_min} ns  "
              f"all=[{e2e_str}]  {status}")

        row = {
            "crossings":       n_cross,
            "sequence":        summary,
            "status":          status,
            "e2e_ns_median":   e2e_median,
            "e2e_ns_min":      e2e_min,
            "e2e_ns_all":      e2e_str,
            "build_time_min":  round(t_hw / 60, 1),
        }
        results.append(row)
        with open(os.path.join(run_dir, "results.json"), "w") as f:
            json.dump(row, f, indent=2)

    # Restore gen.py
    with open(GEN_PY, "w") as f:
        f.write(gen_original)
    print("\n[gen.py restored]")

    # CSV
    csv_path = os.path.join(RUNS_DIR, "sweep_qemu.csv")
    if results:
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=results[0].keys())
            w.writeheader()
            w.writerows(results)
        print(f"CSV: {csv_path}")

    # Summary table
    print(f"\n{'─'*75}")
    print(f"  {'Cross':>5}  {'Sequence':<32}  {'Median ns':>10}  {'Min ns':>8}  Status")
    print(f"{'─'*75}")
    for r in results:
        med = f"{r['e2e_ns_median']}" if r["e2e_ns_median"] != "" else "—"
        mn  = f"{r['e2e_ns_min']}"    if r["e2e_ns_min"]    != "" else "—"
        print(f"  {r['crossings']:>5}  {r['sequence']:<32}  {med:>10}  {mn:>8}  {r['status']}")
    print(f"{'─'*75}")


if __name__ == "__main__":
    main()
