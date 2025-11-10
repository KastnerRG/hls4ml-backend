#!/usr/bin/env python

import argparse
import numpy as np
import os, glob, shutil, subprocess
from framework import *


if __name__ == "__main__":

    ap = argparse.ArgumentParser(description="Run AIE dense kernel sim & check.")
    ap.add_argument("--dtype",   type=str, default="i8", help="dtype: i8 or i16 (default: i8)")
    ap.add_argument("--batch",   "-b", type=int, default=4, help="Batch size (default: 4)")
    ap.add_argument("--inputs",  "-i", type=int, default=128, help="Number of inputs/features (default: 128)")
    ap.add_argument("--outputs", "-o", type=int, default=128, help="Number of outputs (default: 128)")
    ap.add_argument("--dataflow", "-d", type=str, default="stream", help="Dataflow type: stream or window (default: stream)")
    ap.add_argument("--iterations", "-t", type=int, default=1, help="Number of iterations (default: 1)")
    args = ap.parse_args()
    BATCH, INPUTS, OUTPUTS, dtype, dataflow, iterations = args.batch, args.inputs, args.outputs, args.dtype, args.dataflow, args.iterations

    if dtype == 'i8':
        m_tile, k_tile, n_tile = 2, 8, 8
    elif dtype == 'i16':
        m_tile, k_tile, n_tile = 2, 4, 8

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

    x0 = np.random.randint(0, 128, size=(BATCH,INPUTS), dtype=TY_DICT[dtype]['np'])
    model = Sequential(iterations=iterations, dtype=dtype, dataflow=dataflow)
    model.add(Dense(N=OUTPUTS, shift=5, relu=True, m_tile=m_tile, k_tile=k_tile, n_tile=n_tile, dtype=dtype, dataflow=dataflow))

    # Build, emit code, and get reference
    y_ref_final = model.build_and_emit(x0)

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
        print(f"\n\n Success: Outputs match ({out_sim.shape})")
    else:
        print("\n\nError: Output does not match\n")
        print(f"Simulation Output ({out_sim.shape}):\n{out_sim}\n")
        print(f"Expected output ({out_ref.shape}):\n{out_ref}\n")
