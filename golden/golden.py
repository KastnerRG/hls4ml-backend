import numpy as np
import os

# Constants
NUM_LAYERS = 1
ITERATIONS = 1

m, k, n = 2,8,8 #4,8,4

with open("aie/include.h", "w") as f:
    f.write(f"""
#ifndef FUNCTION_INCLUDES_H
#define FUNCTION_INCLUDES_H
#define N_LAYERS {NUM_LAYERS}
#define ITERATIONS {ITERATIONS}
#define m {m}
#define k {k}
#define n {n}
#endif
""")

# Create output directory
os.makedirs("data", exist_ok=True)

def tile_matrix(matrix, row_tiles, col_tiles):
    rows, cols = matrix.shape
    assert rows % row_tiles == 0 and cols % col_tiles == 0, "Matrix must be divisible by block sizes"
    reshaped = matrix.reshape(rows // row_tiles, row_tiles, cols // col_tiles, col_tiles)
    transposed = reshaped.transpose(0, 2, 1, 3)
    tiled = transposed.reshape(-1)
    return tiled


def process_layer(idx, layer):

    matA = layer["x"]
    matB = layer["k"]
    matC = layer["a"]
    
    ''' Generate and save weights'''

    matB_tiled = tile_matrix(matB, k, n)
    np.savetxt(f"data/matB{idx}.txt", matB, fmt="%d")

    with open("aie/weights.h", 'a') as f:
        array_str = ', '.join(str(x) for x in matB_tiled)
        f.write(f"""const int8_t matB{idx} [{matB_tiled.size}] = {{ {array_str} }};\n""")

    ''' Generate input and output matrices '''
    with open(f"data/matA{idx}.txt", "w") as f_a, open(f"data/matC{idx}.txt", "w") as f_c:
        for i in range(ITERATIONS):

            np.savetxt(f"data/orig_matA{idx}_{i}.txt", matA, fmt="%d")
            np.savetxt(f"data/orig_matC{idx}_{i}.txt", matC, fmt="%d")

            matA_tiled = tile_matrix(matA, m, k)
            matC_tiled = tile_matrix(matC, m, n)

            for i, val in enumerate(matA_tiled):
                f_a.write(f"{val}")
                f_a.write("\n" if i % 16 == 15 else " ")

            for i, val in enumerate(matC_tiled):
                f_c.write(f"{val}")
                f_c.write("\n" if i % 16 == 15 else " ")



if __name__ == "__main__":

    layers = []

    file_path = "aie/weights.h"
    if os.path.exists(file_path):
        os.remove(file_path)

    layers += [{}]
    layers[0]["x"] = np.random.randint(0, 128, size=(16, 32), dtype=np.int8)
    layers[0]["k"] = np.random.randint(0, 128, size=(32, 32), dtype=np.int8)
    layers[0]["y"] = np.matmul(layers[0]["x"].astype(np.int32), layers[0]["k"].astype(np.int32))
    layers[0]["a"] = np.maximum(0, layers[0]["y"].astype(np.int8))  # ReLU

    process_layer(0, layers[0])

    # layers += [{}]
    # layers[1]['x'] = layers[0]["a"]
    # layers[1]["k"] = np.random.randint(0, 128, size=(32, 16), dtype=np.int8)
    # layers[1]["y"] = np.matmul(layers[1]["x"].astype(np.int32), layers[1]["k"].astype(np.int32))
    # layers[1]["a"] = np.maximum(0, layers[1]["y"].astype(np.int8))  # ReLU

    # process_layer(1, layers[1])

    # 1. model.cc - each layer as function

    with open("aie/model.cc", "w") as f:
        f.write("""
#include <adf.h>
#include "aie_api/aie.hpp"
#include "aie_api/aie_adf.hpp"
#include "kernels.h"
#include "weights.h"
""")
        for i in range (NUM_LAYERS):
            f.write(f"void f{i}(input_window_int8* __restrict matA, output_window_int8 * __restrict matC) {{ gemm<{m}, {k}, {n}, {layers[i]['x'].shape[0]}, {layers[i]['x'].shape[1]}, {layers[i]['k'].shape[1]}, 0>(matA, matC, matB{i}); }}\n")

  

    # 3. model.h - Function prototypes

    with open("aie/model.h", "w") as f:
        for i in range (NUM_LAYERS):
            f.write(f"void f{i}( input_window_int8  * __restrict, output_window_int8 * __restrict matC);\n")


    # 3. layer_graph.h - create and connect layers

    with open("aie/layer_graph.h", "w") as f:
        f.write(f'A = input_plio::create(plio_128_bits, "data/matA0.txt");\n')
        f.write(f'C = output_plio::create(plio_128_bits, "data/matC{NUM_LAYERS-1}.txt");\n')

        for i in range (NUM_LAYERS):
            f.write(f"layers[{i}] = kernel::create(f{i});\n")
        
        bytes = layers[0]['x'].size * layers[0]['x'].itemsize
        f.write(f"connect<window<{bytes}>>(A.out[0], layers[0].in[0]);\n")
        for i in range (NUM_LAYERS):
            bytes = layers[i]['a'].size * layers[i]['a'].itemsize
            if i == NUM_LAYERS-1:
                f.write(f"connect<window<{bytes}>>(layers[{i}].out[0], C.in[0]);\n")
            else:
                f.write(f"connect<window<{bytes}>>(layers[{i}].out[0], layers[{i+1}].in[0]);\n")

