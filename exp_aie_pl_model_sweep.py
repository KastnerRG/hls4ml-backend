#!/usr/bin/env python3
"""
exp_aie_pl_model_sweep.py
Sweep PL-AIE crossings for a 16-layer int8 dense NN.
  - 16 layers total: 8 PL + 8 AIE, each (batch=8, 48→48), ReLU
  - First and last layers always PL
  - x-axis: number of PL-AIE crossings (2, 4, 6, 8, 10, 12, 14)
  - For each config: make clean → make sim (AIE sim + HLS synthesis in parallel)
  - Collects AIE + PL latencies (actual, from aiesimulator and HLS csynth reports)
  - Each run saved to runs/16_layers__each_8_48_48__{crossings}/

Usage:
    cd /path/to/hls4ml-backend
    python exp_aie_pl_model_sweep.py
"""
import glob, os, re, json, csv, shutil, subprocess, sys, time
from collections import defaultdict

# ── Config ────────────────────────────────────────────────────────────────────
AIE_DIR   = os.path.join(os.path.dirname(os.path.abspath(__file__)), "aie_pl")
RUNS_DIR  = os.path.join(AIE_DIR, "runs")
GEN_PY    = os.path.join(AIE_DIR, "gen.py")

BATCH        = 8
FEAT         = 48           # n_in = n_out = 48 for every layer
N_LAYERS     = 16           # total layers
N_PL         = 8            # PL layers (fixed half)
N_AIE        = 8            # AIE layers (fixed half)
REUSE_FACTOR = 1            # HLS reuse factor
SHIFT        = 0            # right-shift for fixed-point accumulator

CROSSINGS = [2, 4, 6, 8, 10, 12, 14]

AIE_GHZ = 1.25
PL_MHZ  = 312.5

# ── Layer assignment ──────────────────────────────────────────────────────────

def distribute(total, groups):
    """Distribute `total` items into `groups` groups, front-heavy."""
    base, rem = divmod(total, groups)
    return [base + (1 if i < rem else 0) for i in range(groups)]


def make_layers(n_crossings):
    """
    Build LAYERS list for a given crossing count.

    Constraints:
      - 16 layers total, 8 PL + 8 AIE
      - First and last layer always PL
      - n_crossings must be even; pattern: PL→AIE→PL→...→PL

    n_crossings = 2*(k) where k = number of PL→AIE transitions.
    That means k+1 PL segments and k AIE segments.

    Examples:
      n_crossings=2  → PL(4)→AIE(8)→PL(4)           [k=1]
      n_crossings=4  → PL(3)→AIE(4)→PL(3)→AIE(4)→PL(2)  [k=2]
      n_crossings=14 → PL(1)×8 interleaved with AIE(1)×7  [k=7]
    """
    assert n_crossings % 2 == 0, "n_crossings must be even"
    k = n_crossings // 2              # number of AIE segments (= PL→AIE transitions)
    n_pl_segs  = k + 1                # PL segments
    n_aie_segs = k                    # AIE segments

    pl_sizes  = distribute(N_PL,  n_pl_segs)
    aie_sizes = distribute(N_AIE, n_aie_segs)

    layers = []
    for i in range(n_pl_segs):
        for _ in range(pl_sizes[i]):
            layers.append(("pl",  FEAT, FEAT))
        if i < n_aie_segs:
            for _ in range(aie_sizes[i]):
                layers.append(("aie", FEAT, FEAT))

    assert len(layers) == N_LAYERS, \
        f"Expected {N_LAYERS} layers, got {len(layers)}"
    assert layers[0][0]  == "pl", "First layer must be PL"
    assert layers[-1][0] == "pl", "Last layer must be PL"
    assert sum(1 for t,_,_ in layers if t=="pl")  == N_PL
    assert sum(1 for t,_,_ in layers if t=="aie") == N_AIE

    return layers


def seg_summary(layers):
    """E.g. 'PL×4 → AIE×8 → PL×4'"""
    segs = []
    cur = layers[0][0]; cnt = 1
    for t, _, _ in layers[1:]:
        if t == cur:
            cnt += 1
        else:
            segs.append(f"{cur.upper()}×{cnt}"); cur = t; cnt = 1
    segs.append(f"{cur.upper()}×{cnt}")
    return " → ".join(segs)


# ── gen.py patching ───────────────────────────────────────────────────────────

def patch_gen_layers(layers):
    """Overwrite the LAYERS = [...] block in gen.py."""
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
}


def run_make(target, log_path):
    env = os.environ.copy()
    env.update(MAKE_ENV)
    t0 = time.time()
    proc = subprocess.run(
        ["make", target],
        cwd=AIE_DIR, env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True,
    )
    elapsed = time.time() - t0
    with open(log_path, "w") as f:
        f.write(proc.stdout)
    return proc.returncode, proc.stdout, elapsed


# ── Latency extraction ────────────────────────────────────────────────────────

def collect_latency():
    """
    Collect actual latency numbers from aiesimulator and HLS csynth reports.
    Returns dict with:
      aie_cycles   – sum of all g*_latency.json  (cycle-accurate from aiesimulator)
      pl_cycles    – sum of all pl_group*_csynth.rpt latency (from HLS scheduler)
      aie_ns, pl_ns, total_ns
      pl_source    – "HLS report" or "est."
      n_aie_groups, n_pl_groups
    """
    # AIE: sum all group latency files.
    # Hardware perf counters are limited (~4 pairs on VEK280); groups beyond that
    # return 0. Substitute zeros with the mean of valid readings for groups of the
    # same layer count.
    cfg_path = os.path.join(AIE_DIR, "data", "config.json")
    aie_group_sizes = json.load(open(cfg_path)).get("aie_group_sizes", []) \
        if os.path.exists(cfg_path) else []

    raw = []
    for lat_f in sorted(glob.glob(
            os.path.join(AIE_DIR, "aiesimulator_output", "data", "g*_latency.json"))):
        raw.append(json.load(open(lat_f))["cycles"])
    n_aie = len(raw)

    # Build mean cycles per layer-count from non-zero readings
    valid_by_size = defaultdict(list)
    for gi, cyc in enumerate(raw):
        if cyc > 0 and gi < len(aie_group_sizes):
            valid_by_size[aie_group_sizes[gi]].append(cyc)
    mean_by_size = {sz: sum(v)/len(v) for sz, v in valid_by_size.items()}

    extrap_map = {}  # gi -> (mean_val, valid_readings)
    fixed = []
    for gi, cyc in enumerate(raw):
        if cyc == 0 and gi < len(aie_group_sizes):
            sz = aie_group_sizes[gi]
            mean_val = int(round(mean_by_size.get(sz, 0)))
            extrap_map[gi] = (mean_val, list(valid_by_size.get(sz, [])))
            cyc = mean_val
        fixed.append(cyc)

    aie_cycles = sum(fixed)

    # PL: sum HLS synthesis reports (actual scheduled latency from v++ -c --mode hls)
    pl_cycles = 0
    n_pl = 0
    for rpt in sorted(glob.glob(
            os.path.join(AIE_DIR, "pl_group*/hls/syn/report/pl_group*_csynth.rpt"))):
        if not re.fullmatch(r'pl_group\d+_csynth\.rpt', os.path.basename(rpt)):
            continue
        text = open(rpt).read()
        # Latency summary row: | lat_min | lat_max | interval_min | interval_max | type |
        m = re.search(r'\|\s*\d+\|\s*(\d+)\|[^|]+\|[^|]+\|\s*\d+\|\s*\d+\|', text)
        if m:
            pl_cycles += int(m.group(1))
            n_pl += 1

    pl_source = "HLS report" if pl_cycles else "est."
    if not pl_cycles:
        cfg = json.load(open(os.path.join(AIE_DIR, "data", "config.json")))
        pl_cycles = cfg.get("pl_cycles_est", 0)

    aie_ns   = aie_cycles / AIE_GHZ if aie_cycles else 0
    pl_ns    = pl_cycles / (PL_MHZ * 1e6) * 1e9 if pl_cycles else 0
    total_ns = aie_ns + pl_ns

    # Human-readable summary of per-group readings for CSV
    # Format: "g0:3364 g1:*889(mean of [871,881,898]) g2:881 ..."
    parts = []
    for gi, cyc in enumerate(fixed):
        if gi in extrap_map:
            mean_val, readings = extrap_map[gi]
            parts.append(f"g{gi}:*{mean_val}(mean of {readings})")
        else:
            parts.append(f"g{gi}:{raw[gi]}")
    aie_group_detail = " | ".join(parts)

    return dict(
        aie_cycles=aie_cycles, pl_cycles=pl_cycles,
        aie_ns=aie_ns, pl_ns=pl_ns, total_ns=total_ns,
        pl_source=pl_source,
        n_aie_groups=n_aie, n_pl_groups=n_pl,
        aie_group_detail=aie_group_detail,
    )


def run_check_py():
    """Run check.py for correctness verification; return (passed, output)."""
    proc = subprocess.run(
        [sys.executable, os.path.join(AIE_DIR, "check.py")],
        cwd=AIE_DIR,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    return proc.returncode == 0, proc.stdout


# ── Artifact archiving ────────────────────────────────────────────────────────

def archive_run(run_dir):
    """Copy per-run artifacts into run_dir for a permanent record.

    Directory structure mirrors the aie_pl/ layout so that vitis_analyzer can
    be launched directly from run_dir:
        cd run_dir && vitis_analyzer aiesimulator_output/default.aierun_summary
    """
    # ── Vitis analyzer: Work/ and aiesimulator_output/ ───────────────────────
    for dirname in ("Work", "aiesimulator_output"):
        src = os.path.join(AIE_DIR, dirname)
        dst = os.path.join(run_dir, dirname)
        if os.path.isdir(src):
            if os.path.exists(dst):
                shutil.rmtree(dst)
            shutil.copytree(src, dst)

    # ── Top-level analyzer / report files ─────────────────────────────────────
    for name in ("Map_Report.csv", "sol.db",
                 "AIECompiler.log", "AIESimulator.log",
                 "aie_compile.log", "aie_sim.log",
                 "aiesimulator_debug.log", "diag_report.log"):
        src = os.path.join(AIE_DIR, name)
        if os.path.exists(src):
            shutil.copy2(src, run_dir)

    # ── HLS csynth reports ────────────────────────────────────────────────────
    for rpt in glob.glob(os.path.join(AIE_DIR, "pl_group*/hls/syn/report/pl_group*_csynth.rpt")):
        shutil.copy2(rpt, run_dir)

    # ── config.json + latency JSONs ───────────────────────────────────────────
    cfg_src = os.path.join(AIE_DIR, "data", "config.json")
    if os.path.exists(cfg_src):
        shutil.copy2(cfg_src, run_dir)
    for f in glob.glob(os.path.join(AIE_DIR, "aiesimulator_output", "data", "g*_latency.json")):
        shutil.copy2(f, run_dir)


# ── Main sweep ────────────────────────────────────────────────────────────────

def main():
    os.makedirs(RUNS_DIR, exist_ok=True)
    results = []

    with open(GEN_PY) as f:
        gen_original = f.read()

    print(f"{'='*70}")
    print(f"  PL-AIE Crossing Sweep")
    print(f"  Model: {N_LAYERS} layers × (batch={BATCH}, {FEAT}→{FEAT}) int8, RF={REUSE_FACTOR}")
    print(f"  Split: {N_PL} PL + {N_AIE} AIE, first+last always PL")
    print(f"  Crossings: {CROSSINGS}")
    print(f"  Latency: AIE from aiesimulator [cycle-exact] + PL from HLS csynth [HLS report]")
    print(f"{'='*70}")

    for n_cross in CROSSINGS:
        run_dir = os.path.join(RUNS_DIR,
                               f"16_layers__each_{BATCH}_{FEAT}_{FEAT}__{n_cross}")
        os.makedirs(run_dir, exist_ok=True)

        layers = make_layers(n_cross)
        k = n_cross // 2
        summary = seg_summary(layers)

        print(f"\n[crossings={n_cross}]  {k} AIE segment(s), {k+1} PL segment(s)")
        print(f"  Sequence: {summary}")

        # Save layer config for this run
        with open(os.path.join(run_dir, "layers.json"), "w") as f:
            json.dump({"n_crossings": n_cross, "n_pl": N_PL, "n_aie": N_AIE,
                       "batch": BATCH, "feat": FEAT, "reuse_factor": REUSE_FACTOR,
                       "sequence": summary, "layers": layers}, f, indent=2)

        patch_gen_layers(layers)

        # make clean
        print("  make clean ...", flush=True)
        run_make("clean", os.path.join(run_dir, "clean.log"))

        # make sim: runs AIE compile + aiesimulator + HLS synthesis in parallel,
        # then check.py. Both AIE and PL latencies come from actual simulation/synthesis.
        print("  make sim  ...", flush=True)
        rc_sim, out_sim, t_sim = run_make("sim", os.path.join(run_dir, "sim.log"))
        print(f"  make sim  done  ({t_sim/60:.1f} min, rc={rc_sim})")

        if rc_sim != 0:
            print("  ERROR: make sim failed")
            print("  Last 800 chars:\n" + out_sim[-800:])
            results.append({
                "crossings": n_cross, "sequence": summary, "status": "FAIL",
                "aie_cycles": 0, "pl_cycles": 0,
                "aie_ns": 0, "pl_ns": 0, "total_ns": 0,
                "pl_source": "", "n_aie_groups": 0, "n_pl_groups": 0,
                "aie_group_detail": "",
                "sim_time_min": round(t_sim / 60, 1),
            })
            continue

        # Correctness check
        passed, check_out = run_check_py()
        with open(os.path.join(run_dir, "check.log"), "w") as f:
            f.write(check_out)

        # Collect latencies directly from artifacts
        lat = collect_latency()

        print(f"  AIE : {lat['aie_cycles']} cyc  ({lat['aie_ns']:.1f} ns)  "
              f"[{lat['n_aie_groups']} group(s), aiesimulator]")
        print(f"  PL  : {lat['pl_cycles']} cyc  ({lat['pl_ns']:.1f} ns)  "
              f"[{lat['n_pl_groups']} group(s), {lat['pl_source']}]")
        print(f"  Total: {lat['total_ns']:.1f} ns  {'✓' if passed else '✗ (mismatch)'}")

        # Archive artifacts into run folder
        archive_run(run_dir)

        row = {
            "crossings":        n_cross,
            "sequence":         summary,
            "status":           "pass" if passed else "mismatch",
            "aie_cycles":       lat["aie_cycles"],
            "pl_cycles":        lat["pl_cycles"],
            "aie_ns":           round(lat["aie_ns"],   1),
            "pl_ns":            round(lat["pl_ns"],     1),
            "total_ns":         round(lat["total_ns"],  1),
            "pl_source":        lat["pl_source"],
            "n_aie_groups":     lat["n_aie_groups"],
            "n_pl_groups":      lat["n_pl_groups"],
            "aie_group_detail": lat["aie_group_detail"],
            "sim_time_min":     round(t_sim / 60, 1),
        }
        results.append(row)
        with open(os.path.join(run_dir, "results.json"), "w") as f:
            json.dump(row, f, indent=2)

    # Restore gen.py
    with open(GEN_PY, "w") as f:
        f.write(gen_original)
    print("\n[gen.py restored]")

    # CSV
    csv_path = os.path.join(RUNS_DIR, "sweep_results.csv")
    if results:
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=results[0].keys())
            w.writeheader()
            w.writerows(results)
        print(f"CSV: {csv_path}")

    # Plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        valid = [r for r in results if r["total_ns"] > 0]
        if valid:
            xs  = [r["crossings"] for r in valid]
            fig, ax = plt.subplots(figsize=(9, 5))
            ax.plot(xs, [r["total_ns"] for r in valid],
                    "o-",  color="C0", lw=2, ms=8, label="Total")
            ax.plot(xs, [r["aie_ns"]   for r in valid],
                    "s--", color="C1", lw=1.5, ms=6, label="AIE [aiesimulator]")
            ax.plot(xs, [r["pl_ns"]    for r in valid],
                    "^--", color="C2", lw=1.5, ms=6, label="PL [HLS report]")
            ax.set_xlabel("Number of PL↔AIE Crossings", fontsize=12)
            ax.set_ylabel("Latency (ns)", fontsize=12)
            ax.set_title(
                f"16-layer Dense NN: Latency vs PL-AIE Crossings\n"
                f"(batch={BATCH}, {FEAT}×{FEAT} int8, RF={REUSE_FACTOR}, VEK280)",
                fontsize=12)
            ax.set_xticks(CROSSINGS)
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plot_path = os.path.join(RUNS_DIR, "latency_vs_crossings.png")
            plt.savefig(plot_path, dpi=150)
            plt.close()
            print(f"Plot: {plot_path}")
    except ImportError:
        print("matplotlib not available — skipping plot")

    # Summary table
    print(f"\n{'─'*80}")
    print(f"  {'Cross':>5}  {'Sequence':<30}  {'AIE ns':>8}  {'PL ns':>8}  "
          f"{'Total ns':>10}  {'PL src':>10}  Status")
    print(f"{'─'*80}")
    for r in results:
        print(f"  {r['crossings']:>5}  {r['sequence']:<30}  "
              f"{r['aie_ns']:>8.1f}  {r['pl_ns']:>8.1f}  "
              f"{r['total_ns']:>10.1f}  {r['pl_source']:>10}  {r['status']}")
    print(f"{'─'*80}")


if __name__ == "__main__":
    main()
