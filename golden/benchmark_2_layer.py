#!/usr/bin/env python3
import os, sys, csv, re, subprocess, itertools
from pathlib import Path
from pprint import pprint


# DTYPES    = ["i8"]
# BATCHES   = [4]
# INPUTS    = [16, 32, 64, 128, 256, 512, 1024]
# OUTPUTS   = [16, 32, 64, 128, 256, 512, 1024]
# DATAFLOWS = ["stream", "window"]
# combos = list(itertools.product(DTYPES, BATCHES, INPUTS, OUTPUTS, DATAFLOWS))

input_output = (
    [32  , 64], # 2K
    # [48  , 64], # 3K
    [64  , 64], # 4K
    # [80  , 64], # 5K
    [96  , 64], # 6K
    # [112 , 64],# 7K
    [128 , 64],# 8K
    # [144 , 64],# 9K
    [160 , 64],# 10K
    # [176 , 64],# 11K
    [192 , 64],# 12K
    # [208 , 64],# 13K
    [224 , 64],# 14K
    # [240 , 64],# 15K
    [256 , 64],# 16K
    # [512 , 64],# 32K
)


result_dir = f'vitis_work_benchmark_2_layer'

combos = []
for io in input_output:
    combos += [['i8', 4, *io, 'stream']]


ITERATIONS = 10

PYTHON_EXE = sys.executable  # or set to "python3"
CSV_PATH   = "dense_results.csv"
APPEND     = True
VERBOSE    = False

# Examples matched:
#   Graph Latency (first->first): 145 cycles, 120.8 ns
#   GRAPH LATENCY ... : 882 cycles, 735.0 ns
LAT_RE = re.compile(
    r"Graph\s+Latency.*?:\s*\d+\s*cycles,\s*([0-9.]+)\s*ns",
    re.IGNORECASE
)

# Throughput table lines, e.g.:
#   mygraph_AIE_IN  | IN  | 5714.285714 MBps
#   mygraph_AIE_OUT | OUT | 1538.461538 MBps
THR_RE = re.compile(
    r"^\s*(?P<port>\S+)\s*\|\s*(?P<type>IN|OUT)\s*\|\s*(?P<val>[0-9.]+)\s*MBps\s*$",
    re.IGNORECASE | re.MULTILINE
)

SUCCESS_TOKEN = "Success: Outputs match"

def run_cmd(cmd):
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

def extract_throughputs(stdout: str):
    """
    Returns (in_MBps, out_MBps) as floats or (None, None) if not found.
    If multiple lines exist, last occurrence per type wins.
    """
    in_mb, out_mb = None, None
    for m in THR_RE.finditer(stdout):
        typ = m.group("type").upper()
        val = float(m.group("val"))
        if typ == "IN":
            in_mb = val
        elif typ == "OUT":
            out_mb = val
    return in_mb, out_mb

def _flush_now(f):
    f.flush()
    try:
        os.fsync(f.fileno())
    except OSError:
        pass

def open_csv(path: Path, append: bool):
    exists = path.exists() and append
    f = path.open("a" if append else "w", newline="", buffering=1)  # line-buffered
    w = csv.writer(f)
    if not exists:
        w.writerow([
            "dataflow", "dtype", "batch", "inputs", "outputs",
            "latency_ns", "in_MBps", "out_MBps", "failure"
        ])
        _flush_now(f)
    return f, w

def write_row_live(writer, f, row):
    writer.writerow(row)
    _flush_now(f)

def main():
    out_path = Path(CSV_PATH).resolve()
    f, writer = open_csv(out_path, append=APPEND)

    os.makedirs(result_dir, exist_ok=True)

    try:
        for dtype, batch, ins, outs, dataflow in combos:
            plio = ins//32
            cmd = [
                PYTHON_EXE, 'run_workload.py',
                "--dtype", dtype, 
                "-b", str(batch),
                "-i", str(ins), 
                "-o", str(ins), #in=out
                "-d", dataflow, 
                "-t", f"{ITERATIONS}", 
                "-w", "dense_in_out",
                '-p', str(plio),
                '-q', str(plio),
                '-r', result_dir
            ]
            print(f"\n▶ Running {dataflow}: {os.path.basename('run_workload.py')} {' '.join(cmd[2:])}")
            rc, stdout = run_cmd(cmd)
            if VERBOSE:
                print(stdout)

            lat_ns = extract_latency_ns(stdout)
            in_thr, out_thr = extract_throughputs(stdout)

            ok = (rc == 0) and (SUCCESS_TOKEN in stdout)
            failure = not ok

            latency_field = f"{lat_ns:.3f}" if lat_ns is not None else ""
            in_field  = f"{in_thr:.6f}"  if in_thr  is not None else ""
            out_field = f"{out_thr:.6f}" if out_thr is not None else ""

            write_row_live(
                writer, f,
                [dataflow, dtype, batch, ins, outs, latency_field, in_field, out_field, "yes" if failure else "no"]
            )

            if failure:
                reason = []
                if rc != 0: reason.append(f"rc={rc}")
                if SUCCESS_TOKEN not in stdout: reason.append("no success token")
                if lat_ns is None: reason.append("no latency")
                if in_thr is None: reason.append("no IN throughput")
                if out_thr is None: reason.append("no OUT throughput")
                print(f"  ✖ Logged failure: {', '.join(reason) if reason else 'unknown'}")
            else:
                print(
                    f"  ✓ Logged success: dataflow={dataflow} dtype={dtype} "
                    f"B={batch} I={ins} O={outs} latency={latency_field} ns "
                    f"in={in_field} MBps out={out_field} MBps"
                )

        print(f"\nDone. Results appended to: {out_path}")
    finally:
        f.close()

if __name__ == "__main__":
    main()
