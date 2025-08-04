import numpy as np
import os

# Constants
ITERATIONS = 1
M = 16
K = 32
N = 16
m = 4
k = 8
n = 4

with open("aie/include.h", "w") as f:
    f.write(f"""
#ifndef FUNCTION_INCLUDES_H
#define FUNCTION_INCLUDES_H
#define N_LAYERS 1
#define SHIFT 0
#define ITERATIONS {ITERATIONS}
#define M {M}
#define K {K}
#define N {N}
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


def process_layer(idx, matA, matB, matC):
    
    ''' Generate and save weights'''

    matB_tiled = tile_matrix(matB, k, n)
    np.savetxt(f"data/matB{idx}.txt", matB, fmt="%d")

    with open("aie/weights.h", 'w') as f:
        array_str = ', '.join(str(x) for x in matB_tiled)
        f.write(f"""const int8_t matB0 [{matB_tiled.size}] = {{ {array_str} }};\n""")

    ''' Generate input and output matrices '''
    with open(f"data/matA{idx}.txt", "w") as f_a, open(f"data/matC{idx}.txt", "w") as f_c:
        for i in range(ITERATIONS):

            np.savetxt(f"data/matA{idx}_{i}.txt", matA, fmt="%d")
            np.savetxt(f"data/matC{idx}_{i}.txt", matC, fmt="%d")

            matA_tiled = tile_matrix(matA, m, k)
            matC_tiled = tile_matrix(matC, m, n)

            for i, val in enumerate(matA_tiled):
                f_a.write(f"{val}")
                f_a.write("\n" if i % 16 == 15 else " ")

            for i, val in enumerate(matC_tiled):
                f_c.write(f"{val}")
                f_c.write("\n" if i % 16 == 15 else " ")



if __name__ == "__main__":

    matA_ori = np.random.randint(0, 128, size=(M, K), dtype=np.int8)
    matB_ori = np.random.randint(0, 128, size=(K, N), dtype=np.int8)
    matC_ori = np.matmul(matA_ori.astype(np.int32), matB_ori.astype(np.int32))  # Promote to int32 for accumulation
    matC_ori = np.maximum(0, matC_ori.astype(np.int8))  # Apply ReLU activation

    process_layer(0, matA_ori, matB_ori, matC_ori)

