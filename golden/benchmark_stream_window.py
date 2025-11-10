#!/usr/bin/env python3
import os, sys, csv, re, subprocess, itertools
from pathlib import Path

# -------------------------
# Edit these lists as needed
# -------------------------
DTYPES   = ["i8"]
BATCHES  = [4]
# INPUTS   = [16, 32, 48, 64, 80, 96, 112, 128, 256, 512, 1024, 2048]
# OUTPUTS  = [16, 32, 48, 64, 80, 96, 112, 128, 256, 512, 1024, 2048]
INPUTS   = [16]
OUTPUTS  = [16]
DATAFLOWS = ["stream", "window"]

PYTHON_EXE = sys.executable  # or set to "python3"
CSV_PATH   = "dense_results.csv"
APPEND     = True            # <-- append to existing CSV
VERBOSE    = False

# Regex to capture latency like: "... : 882 cycles, 735.0 ns"
LAT_RE = re.compile(r"GRAPH LATENCY.*?:\s*\d+\s*cycles,\s*([0-9.]+)\s*ns", re.IGNORECASE)
SUCCESS_TOKEN = "Success: Outputs match"

def run_cmd(cmd):
    # Decode as UTF-8 but *never* fail: replace bad bytes so regex still works
    p = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return p.returncode, p.stdout

def extract_latency_ns(stdout: str):
    matches = LAT_RE.findall(stdout)
    return float(matches[-1]) if matches else None

def _flush_now(f):
    f.flush()
    try:
        os.fsync(f.fileno())
    except OSError:
        pass

def open_csv(path: Path, append: bool):
    exists = path.exists()
    f = path.open("a" if append else "w", newline="", buffering=1)  # line-buffered
    w = csv.writer(f)
    if not exists:  # write header only if new file
        w.writerow(["stream/dense", "dtype", "batch", "inputs", "outputs", "latency_ns", "failiure"])
        _flush_now(f)
    return f, w

def write_row_live(writer, f, row):
    writer.writerow(row)
    _flush_now(f)

def main():
    combos = list(itertools.product(DTYPES, BATCHES, INPUTS, OUTPUTS, DATAFLOWS))
    out_path = Path(CSV_PATH).resolve()
    f, writer = open_csv(out_path, append=APPEND)

    try:
        for dtype, batch, ins, outs, dataflow in combos:
            cmd = [PYTHON_EXE, 'dense_exp.py', "--dtype", dtype, "-b", str(batch), "-i", str(ins), "-o", str(outs), "-d", dataflow, "--iterations", "4"]
            print(f"\n▶ Running {dataflow}: {os.path.basename('dense_exp.py')} {' '.join(cmd[2:])}")
            rc, stdout = run_cmd(cmd)
            if VERBOSE:
                print(stdout)

            lat_ns = extract_latency_ns(stdout)
            ok = (rc == 0) and (SUCCESS_TOKEN in stdout)
            failure = not ok

            latency_field = f"{lat_ns:.3f}" if lat_ns is not None else ""
            write_row_live(writer, f, [dataflow, dtype, batch, ins, outs, latency_field, "yes" if failure else "no"])

            if failure:
                reason = []
                if rc != 0: reason.append(f"rc={rc}")
                if SUCCESS_TOKEN not in stdout: reason.append("no success token")
                if lat_ns is None: reason.append("no latency")
                print(f"  ✖ Logged failure: {', '.join(reason) if reason else 'unknown'}")
            else:
                print(f"  ✓ Logged success: {dataflow=} {dtype=} B={batch} I={ins} O={outs} latency={latency_field} ns")

        print(f"\nDone. Results appended to: {out_path}")
    finally:
        f.close()

if __name__ == "__main__":
    main()
