import numpy as np
import os

# Constants
M = 16
K = 32
N = 16
m = 4
k = 8
n = 4

# write aie/kernels/include.h
with open("aie/include.h", "w") as f:
    f.write(f"""
#ifndef FUNCTION_INCLUDES_H
#define FUNCTION_INCLUDES_H
#define SHIFT 0
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

def tile_matrix(matrix, row_tiles, col_tiles, dtype=None):
    rows, cols = matrix.shape
    assert rows % row_tiles == 0 and cols % col_tiles == 0, "Matrix must be divisible by block sizes"
    reshaped = matrix.reshape(rows // row_tiles, row_tiles, cols // col_tiles, col_tiles)
    transposed = reshaped.transpose(0, 2, 1, 3)
    tiled = transposed.reshape(-1).astype(dtype)
    return tiled

matB_ori = np.random.randint(0, 128, size=(K, N), dtype=np.int8)
matB_tiled = tile_matrix(matB_ori, k, n, dtype=np.int8)

np.savetxt("data/matB0_ori.txt", matB_ori, fmt="%d")

with open("aie/weights.h", 'w') as f:
    array_str = ', '.join(str(x) for x in matB_tiled)
    f.write(f"""const int8_t matB [{matB_tiled.size}] = {{ {array_str} }};\n""")


with open("data/matB0.txt", "w") as f:
    for _ in range(10):
        for i, val in enumerate(matB_tiled):
            f.write(f"{val}")
            f.write("\n" if i % 16 == 15 else " ")


matA_ori = np.random.randint(0, 128, size=(M, K), dtype=np.int8)
matC_ori = np.matmul(matA_ori.astype(np.int32), matB_ori.astype(np.int32))  # Promote to int32 for accumulation

np.savetxt("data/matA0_ori.txt", matA_ori, fmt="%d")
np.savetxt("data/matC0_ori.txt", matC_ori, fmt="%d")

matA_tiled = tile_matrix(matA_ori, m, k, dtype=np.int8)
matC_tiled = tile_matrix(matC_ori, m, n, dtype=np.int32)

with open("data/matA0.txt", "w") as f:
    for _ in range(10):
        for i, val in enumerate(matA_tiled):
            f.write(f"{val}")
            f.write("\n" if i % 16 == 15 else " ")

with open("data/matC0.txt", "w") as f:
    for _ in range(10):
        for i, val in enumerate(matC_tiled):
            f.write(f"{val}")
            f.write("\n" if i % 4 == 3 else " ")
