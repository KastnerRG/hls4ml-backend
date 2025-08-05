import numpy as np
import os, glob, shutil, subprocess

def tile_matrix(matrix, row_tiles, col_tiles):
    rows, cols = matrix.shape
    assert rows % row_tiles == 0 and cols % col_tiles == 0, "Matrix must be divisible by block sizes"
    reshaped = matrix.reshape(rows // row_tiles, row_tiles, cols // col_tiles, col_tiles)
    transposed = reshaped.transpose(0, 2, 1, 3)
    tiled = transposed.reshape(-1)
    return tiled

def process_layer(idx, layer, iterations):

    matA, matB, matC = layer["x"], layer["k"], layer["y"]

    matB_tiled = tile_matrix(matB, k, n)
    np.savetxt(f"data/matB{idx}.txt", matB, fmt="%d")
    array_str = ', '.join(str(x) for x in matB_tiled)
    with open("aie/weights.h", 'a') as f:
        f.write(f"""const int8_t matB{idx} [{matB_tiled.size}] = {{ {array_str} }};\n""")

    matA_tiled = tile_matrix(matA, m, k)
    matC_tiled = tile_matrix(matC, m, n)
    np.savetxt(f"data/matA{idx}.txt", np.tile(matA_tiled, (iterations, 1)).reshape(-1, 16), fmt="%s", delimiter=" ")
    np.savetxt(f"data/matC{idx}.txt", np.tile(matC_tiled, (iterations, 1)).reshape(-1, 16), fmt="%s", delimiter=" ")



if __name__ == "__main__":
    
    layers = []

    is_relu = True
    shift = 2
    x = np.random.randint(0, 128, size=(16, 32), dtype=np.int8)
    k = np.random.randint(0, 128, size=(32, 32), dtype=np.int8)
    y = np.matmul(x.astype(np.int32), k.astype(np.int32))
    y = (y >> shift).astype(np.int8)
    a = np.maximum(0, y) if is_relu else y
    layers += [{'x': x, 'k': k, 'y': y, 'a': a, 'shift': shift, 'is_relu': is_relu}]

    is_relu = False
    shift = 3
    x = a
    k = np.random.randint(0, 128, size=(32, 64), dtype=np.int8)
    y = np.matmul(x.astype(np.int32), k.astype(np.int32))
    y = (y >> shift).astype(np.int8)
    a = np.maximum(0, y) if is_relu else y
    layers += [{'x': x, 'k': k, 'y': y, 'a': a, 'shift': shift, 'is_relu': is_relu}]

    is_relu = True
    shift = 4
    x = a
    k = np.random.randint(0, 128, size=(64, 32), dtype=np.int8)
    y = np.matmul(x.astype(np.int32), k.astype(np.int32))
    y = (y >> shift).astype(np.int8)
    a = np.maximum(0, y) if is_relu else y
    layers += [{'x': x, 'k': k, 'y': y, 'a': a, 'shift': shift, 'is_relu': is_relu}]

    m, k, n = 2,8,8 # k==n such that output matrix can be fed as input without re-tiling
    iterations = 10

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
#define N_LAYERS {len(layers)}
#define ITERATIONS {iterations}
#endif
    """)

    # 1. Generate weights and input/output matrices

    for i, layer in enumerate(layers):
        process_layer(i, layer, iterations)
    
    tiled_mat = tile_matrix(layers[-1]['a'], m, n)
    np.savetxt("data/out_ref.txt", np.tile(tiled_mat, (iterations, 1)).reshape(-1, 16), fmt="%s", delimiter=" ")

    # 2. model.cc - each layer as function

    with open("aie/model.cc", "w") as f:
        f.write("""
#include <adf.h>
#include "aie_api/aie.hpp"
#include "aie_api/aie_adf.hpp"
#include "kernels.h"
#include "weights.h"
""")
        for i, layer in enumerate(layers):
            t_m = layer['x'].shape[0] // m
            t_k = layer['x'].shape[1] // k
            t_n = layer['k'].shape[1] // n
            shift = layer['shift']
            is_relu = str(layer['is_relu']).lower()
            f.write(f"void f{i}(input_window_int8* __restrict matA, output_window_int8 * __restrict matC) ")
            f.write(f"{{ dense<{m}, {k}, {n}, {t_m}, {t_k}, {t_n}, {shift}, {is_relu}> (matA, matC, matB{i}); }}\n")

    # 3. model.h - Function prototypes

    with open("aie/model.h", "w") as f:
        for i in range (len(layers)):
            f.write(f"void f{i}( input_window_int8  * __restrict, output_window_int8 * __restrict matC);\n")

    # 4. layer_graph.h - create and connect layers

    with open("aie/layer_graph.h", "w") as f:
        f.write(f'A = input_plio::create(plio_128_bits, "data/matA0.txt");\n')
        f.write(f'C = output_plio::create(plio_128_bits, "data/out_sim.txt");\n')

        for i in range (len(layers)):
            f.write(f"layers[{i}] = kernel::create(f{i});\n")
        
        num_bytes = layers[0]['x'].size * layers[0]['x'].itemsize
        f.write(f"connect<window<{num_bytes:>5}>>(A.out[0], layers[0].in[0]);\n")
        for i, layer in enumerate(layers):
            num_bytes = layer['a'].size * layer['a'].itemsize
            out_port = "C" if i == len(layers)-1 else f"layers[{i+1}]"
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
