import numpy as np
import os

# Constants
single_M = 16
single_K = 32
single_N = 16
M_API = 4
K_API = 8
N_API = 4

# Create output directory
os.makedirs("data", exist_ok=True)

# Step 1: Create A and B, compute C
matA_ori = np.random.randint(0, 128, size=(single_M, single_K), dtype=np.int8)
matB_ori = np.random.randint(0, 128, size=(single_K, single_N), dtype=np.int8)
matC_ori = np.matmul(matA_ori.astype(np.int32), matB_ori.astype(np.int32))  # Promote to int32 for accumulation

# Step 2: Save original matrices (row-major)
np.savetxt("data/matA0_ori.txt", matA_ori, fmt="%d")
np.savetxt("data/matB0_ori.txt", matB_ori, fmt="%d")
np.savetxt("data/matC0_ori.txt", matC_ori, fmt="%d")

# Step 3: Blockify and save
def blockify_matrix(matrix, I_API, J_API, dtype=None):
    """
    General-purpose function to blockify a 2D matrix.

    Parameters:
        matrix : np.ndarray of shape (I, J)
        I_API, J_API : int, block sizes
        dtype : optional, output data type (e.g., np.int8, np.int32)

    Returns:
        np.ndarray of shape (I * J,) in flattened blocked format
    """
    I, J = matrix.shape
    assert I % I_API == 0 and J % J_API == 0, "Matrix must be divisible by block sizes"

    if dtype is None:
        dtype = matrix.dtype

    blocked = np.zeros((I * J,), dtype=dtype)
    idx = 0
    for i in range(I // I_API):
        for j in range(J // J_API):
            for i_a in range(I_API):
                for j_a in range(J_API):
                    row = i * I_API + i_a
                    col = j * J_API + j_a
                    blocked[idx] = matrix[row, col]
                    idx += 1
    return blocked


matA_blocked = blockify_matrix(matA_ori, M_API, K_API, dtype=np.int8)
matB_blocked = blockify_matrix(matB_ori, K_API, N_API, dtype=np.int8)
matC_blocked = blockify_matrix(matC_ori, M_API, N_API, dtype=np.int32)

with open("data/matA0.txt", "w") as f:
    for i, val in enumerate(matA_blocked):
        f.write(f"{val}")
        f.write("\n" if i % 16 == 15 else " ")

with open("data/matB0.txt", "w") as f:
    for i, val in enumerate(matB_blocked):
        f.write(f"{val}")
        f.write("\n" if i % 16 == 15 else " ")

with open("data/matC0.txt", "w") as f:
    for i, val in enumerate(matC_blocked):
        f.write(f"{val}")
        f.write("\n" if i % 4 == 3 else " ")
