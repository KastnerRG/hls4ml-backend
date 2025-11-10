import os, sys, csv, re, subprocess, itertools
from pathlib import Path

# -------------------------
# Edit these lists as needed
# -------------------------
SCRIPTS = [("dense",  "dense_window.py"),
           ("stream", "dense_stream.py")]
DTYPES   = ["i8", "i16"]
BATCHES  = [4]
INPUTS   = [16, 32]
OUTPUTS  = [16, 32]

PYTHON_EXE = sys.executable  # or set to "python3"
CSV_PATH   = "dense_results.csv"
APPEND     = False           # True = append, False = overwrite
VERBOSE    = False

# Regex to capture latency like: "... : 882 cycles, 735.0 ns"
LAT_RE = re.compile(r"GRAPH LATENCY.*?:\s*\d+\s*cycles,\s*([0-9.]+)\s*ns", re.IGNORECASE)
SUCCESS_TOKEN = "Success: Outputs match"

def run_cmd(cmd):
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    return p.returncode, p.stdout

def extract_latency_ns(stdout: str):
    matches = LAT_RE.findall(stdout)
    return float(matches[-1]) if matches else None

def open_csv(path: Path, append: bool):
    exists = path.exists()
    f = path.open("a" if append else "w", newline="")
    w = csv.writer(f)
    if not append or not exists:
        w.writerow(["stream/dense", "dtype", "batch", "inputs", "outputs", "latency_ns", "failiure"])
    return f, w

def main():
    combos = list(itertools.product(SCRIPTS, DTYPES, BATCHES, INPUTS, OUTPUTS))
    out_path = Path(CSV_PATH).resolve()
    f, writer = open_csv(out_path, append=APPEND)

    try:
        for (mode, script), dtype, batch, ins, outs in combos:
            cmd = [PYTHON_EXE, script, "--dtype", dtype, "-b", str(batch), "-i", str(ins), "-o", str(outs)]
            print(f"\n▶ Running {mode}: {os.path.basename(script)} {' '.join(cmd[2:])}")
            rc, stdout = run_cmd(cmd)
            if VERBOSE:
                print(stdout)

            lat_ns = extract_latency_ns(stdout)
            ok = (rc == 0) and (SUCCESS_TOKEN in stdout)
            failure = not ok

            # Write a row ALWAYS; empty latency if we couldn't parse it
            latency_field = f"{lat_ns:.3f}" if lat_ns is not None else ""
            writer.writerow([mode, dtype, batch, ins, outs, latency_field, "yes" if failure else "no"])

            if failure:
                reason = []
                if rc != 0: reason.append(f"rc={rc}")
                if SUCCESS_TOKEN not in stdout: reason.append("no success token")
                if lat_ns is None: reason.append("no latency")
                print(f"  ✖ Logged failure: {', '.join(reason) if reason else 'unknown'}")
            else:
                print(f"  ✓ Logged success: {mode=} {dtype=} B={batch} I={ins} O={outs} latency={latency_field} ns")

        print(f"\nDone. Results written to: {out_path}")
    finally:
        f.close()

if __name__ == "__main__":
    main()
