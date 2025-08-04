import numpy as np
import os, glob, shutil, subprocess

def tile_matrix(matrix, row_tiles, col_tiles):
    rows, cols = matrix.shape
    assert rows % row_tiles == 0 and cols % col_tiles == 0, "Matrix must be divisible by block sizes"
    reshaped = matrix.reshape(rows // row_tiles, row_tiles, cols // col_tiles, col_tiles)
    transposed = reshaped.transpose(0, 2, 1, 3)
    tiled = transposed.reshape(-1)
    return tiled

def process_layer(idx, layer):

    matA, matB, matC = layer["x"], layer["k"], layer["y"]

    matB_tiled = tile_matrix(matB, k, n)
    np.savetxt(f"data/matB{idx}.txt", matB, fmt="%d")
    with open("aie/weights.h", 'a') as f:
        array_str = ', '.join(str(x) for x in matB_tiled)
        f.write(f"""const int8_t matB{idx} [{matB_tiled.size}] = {{ {array_str} }};\n""")

    matA_tiled = tile_matrix(matA, m, k)
    with open(f"data/matA{idx}.txt", "w") as f_a:
        for i in range(ITERATIONS):
            np.savetxt(f"data/orig_matA{idx}_{i}.txt", matA, fmt="%d")
            for i, val in enumerate(matA_tiled):
                f_a.write(f"{val}")
                f_a.write("\n" if i % 16 == 15 else " ")

    matC_tiled = tile_matrix(matC, m, n)
    with open(f"data/matC{idx}.txt", "w") as f_c:
        for i in range(ITERATIONS):
            np.savetxt(f"data/orig_matC{idx}_{i}.txt", matC, fmt="%d")
            for i, val in enumerate(matC_tiled):
                f_c.write(f"{val}")
                f_c.write("\n" if i % 16 == 15 else " ")



if __name__ == "__main__":
    
    layers = []

    idx = 0
    layers += [{}]
    layers[idx]["x"] = np.random.randint(0, 128, size=(16, 32), dtype=np.int8)
    layers[idx]["k"] = np.random.randint(0, 128, size=(32, 32), dtype=np.int8)
    layers[idx]['is_relu'] = True
    layers[idx]['shift'] = 2
    layers[idx]["y"] = np.matmul(layers[idx]["x"].astype(np.int32), layers[idx]["k"].astype(np.int32))
    layers[idx]["y"] = (layers[idx]["y"] >> layers[idx]['shift']).astype(np.int8)
    layers[idx]["a"] = np.maximum(0, layers[idx]["y"]) if layers[idx]['is_relu'] else layers[idx]["y"]

    idx = 1
    layers += [{}]
    layers[idx]['x'] = layers[idx-1]["a"]
    layers[idx]["k"] = np.random.randint(0, 128, size=(32, 64), dtype=np.int8)
    layers[idx]['is_relu'] = False
    layers[idx]['shift'] = 3
    layers[idx]["y"] = np.matmul(layers[idx]["x"].astype(np.int32), layers[idx]["k"].astype(np.int32))
    layers[idx]["y"] = (layers[idx]["y"] >> layers[idx]['shift']).astype(np.int8)
    layers[idx]["a"] = np.maximum(0, layers[idx]["y"]) if layers[idx]['is_relu'] else layers[idx]["y"]

    idx = 2
    layers += [{}]
    layers[idx]['x'] = layers[idx-1]["a"]
    layers[idx]["k"] = np.random.randint(0, 128, size=(64, 32), dtype=np.int8)
    layers[idx]['is_relu'] = True
    layers[idx]['shift'] = 4
    layers[idx]["y"] = np.matmul(layers[idx]["x"].astype(np.int32), layers[idx]["k"].astype(np.int32))
    layers[idx]["y"] = (layers[idx]["y"] >> layers[idx]['shift']).astype(np.int8)
    layers[idx]["a"] = np.maximum(0, layers[idx]["y"]) if layers[idx]['is_relu'] else layers[idx]["y"]

    m, k, n = 2,8,8 # k==n such that output matrix can be fed as input without re-tiling
    ITERATIONS = 1
    NUM_LAYERS = len(layers)

    # 0. Do a cleanup

    for path in [
        "data", "aie/include.h", "aie/weights.h", "aie/layer_graph.h", "aie/model.cc", "aie/model.h",
        "*.log", "aiesimulator_output", "Work", ".Xil", 
        ".AIE_SIM_CMD_LINE_OPTIONS", "ISS_RPC_SERVER_PORT",
        "libadf.a", "Map_Report.csv", "pl_sample_counts",
        "plio_throughput_info.json", "sol.db", "aiesim.vcd"
    ]:
        for p in glob.glob(path):
            if os.path.isdir(p):
                shutil.rmtree(p, ignore_errors=True)
            else:
                try: os.remove(p)
                except FileNotFoundError: pass
    
    os.makedirs("data", exist_ok=True)
    
    with open("aie/include.h", "w") as f:
        f.write(f"""
#ifndef FUNCTION_INCLUDES_H
#define FUNCTION_INCLUDES_H
#define N_LAYERS {NUM_LAYERS}
#define ITERATIONS {ITERATIONS}
#endif
    """)

    # 1. Generate weights and input/output matrices

    for i, layer in enumerate(layers):
        process_layer(i, layer)
    
    with open(f"data/out_ref.txt", "w") as f_c:
        for i in range(ITERATIONS):
            for i, val in enumerate(tile_matrix(layers[NUM_LAYERS-1]['a'], m, n)):
                f_c.write(f"{val}")
                f_c.write("\n" if i % 16 == 15 else " ")

    # 2. model.cc - each layer as function

    with open("aie/model.cc", "w") as f:
        f.write("""
#include <adf.h>
#include "aie_api/aie.hpp"
#include "aie_api/aie_adf.hpp"
#include "kernels.h"
#include "weights.h"
""")
        for i in range (NUM_LAYERS):
            tiles_m = layers[i]['x'].shape[0] // m
            tiles_k = layers[i]['x'].shape[1] // k
            tiles_n = layers[i]['k'].shape[1] // n
            shift = layers[i]['shift']
            is_relu = str(layers[i]['is_relu']).lower()
            f.write(f"void f{i}(input_window_int8* __restrict matA, output_window_int8 * __restrict matC) ")
            f.write(f"{{ dense<{m}, {k}, {n}, {tiles_m}, {tiles_k}, {tiles_n}, {shift}, {is_relu}> (matA, matC, matB{i}); }}\n")

    # 3. model.h - Function prototypes

    with open("aie/model.h", "w") as f:
        for i in range (NUM_LAYERS):
            f.write(f"void f{i}( input_window_int8  * __restrict, output_window_int8 * __restrict matC);\n")

    # 4. layer_graph.h - create and connect layers

    with open("aie/layer_graph.h", "w") as f:
        f.write(f'A = input_plio::create(plio_128_bits, "data/matA0.txt");\n')
        f.write(f'C = output_plio::create(plio_128_bits, "data/out_sim.txt");\n')

        for i in range (NUM_LAYERS):
            f.write(f"layers[{i}] = kernel::create(f{i});\n")
        
        num_bytes = layers[0]['x'].size * layers[0]['x'].itemsize
        f.write(f"connect<window<{num_bytes:>5}>>(A.out[0], layers[0].in[0]);\n")
        for i in range (NUM_LAYERS):
            num_bytes = layers[i]['a'].size * layers[i]['a'].itemsize
            out_port = "C" if i == NUM_LAYERS-1 else f"layers[{i+1}]"
            f.write(f"connect<window<{num_bytes:>5}>>(layers[{i}].out[0], {out_port}.in[0]);\n")

    # 5. Run AIE

    subprocess.run(["./run.sh"], check=True)

    # 6. Verify output

    aie_out_path = "aiesimulator_output/data/out_sim.txt"
    assert os.path.exists(aie_out_path), f"Error: Output file {aie_out_path} does not exist."

    with open(aie_out_path, "r") as infile, open("data/out_sim.txt", "w") as outfile:
        for line in infile:
            if not line.startswith("T"):
                outfile.write(line)

    out_sim = np.loadtxt("data/out_sim.txt").astype(np.int32)
    out_ref = np.loadtxt("data/out_ref.txt").astype(np.int32)

    if out_sim.shape == out_ref.shape and np.array_equal(out_sim, out_ref):
        print(f"\n\n Success: Outputs match ({out_sim.shape})\n\n{out_sim}\n\n")
    else:
        print("\n\nError: Output does not match\n")
        print(f"Simulation Output ({out_sim.shape}):\n{out_sim}\n")
        print(f"Expected output ({out_ref.shape}):\n{out_ref}\n")

