#!/usr/bin/env python

import argparse
import importlib
import numpy as np
import os, glob, shutil
import subprocess

if __name__ == "__main__":

    ap = argparse.ArgumentParser(description="Run AIE dense kernel sim & check.")
    ap.add_argument("--dtype",   type=str, default="i8", help="dtype: i8 or i16 (default: i8)")
    ap.add_argument("--batch",   "-b", type=int, default=4, help="Batch size (default: 4)")
    ap.add_argument("--inputs",  "-i", type=int, default=128, help="Number of inputs/features (default: 128)")
    ap.add_argument("--outputs", "-o", type=int, default=16, help="Number of outputs (default: 16)")
    ap.add_argument("--dataflow", "-d", type=str, default="stream", help="Dataflow type: stream or window (default: stream)")
    ap.add_argument("--iterations", "-t", type=int, default=10, help="Number of iterations (default: 1)")
    ap.add_argument("--workload", "-w", type=str, default="dense", help="Workload (default: dense)")
    ap.add_argument("--free", "-f", action="store_true", help="Free running mode")
    ap.add_argument("--input-plios", "-p", type=int, default=4, help="Number of input PLIOs (default: 4)")
    ap.add_argument("--result_dir", "-r", type=str, default="vitis_work", help="Result directory (default: vitis_work)")
    args = ap.parse_args()

    if args.dtype == 'i8':
        m_tile, k_tile, n_tile = 2, 8, 8
    elif args.dtype == 'i16':
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

    module_name = f"workloads.{args.workload}"
    module = importlib.import_module(module_name)

    y_ref_final = module.get_output(
        batch=args.batch, 
        inputs=args.inputs, 
        outputs=args.outputs, 
        dtype=args.dtype, 
        dataflow=args.dataflow, 
        iterations=args.iterations,
        free=args.free,
        m_tile=m_tile, 
        k_tile=k_tile, 
        n_tile=n_tile,
        input_plios=args.input_plios,
    )

    project_dir = f"{args.result_dir}/w{args.workload}_dt{args.dtype}_b{args.batch}_i{args.inputs}_o{args.outputs}_d{args.dataflow}_t{args.iterations}_f{args.free}_p{args.input_plios}"
    if os.path.exists(project_dir):
        shutil.rmtree(project_dir, ignore_errors=True)
    os.makedirs(project_dir, exist_ok=True)
    subprocess.run(["../../run.sh"], check=True, cwd=project_dir)

    # Verify
    aie_out_path = f"{project_dir}/aiesimulator_output/data/out_sim.txt"
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
