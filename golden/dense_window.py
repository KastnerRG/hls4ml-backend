import argparse
import numpy as np
import os, glob, shutil, subprocess
import math

TY_DICT ={
    'i8':{
        'np': np.int8,
        'str': 'int8',
        'per_plio': 16
    },
    'i16':{
        'np': np.int16,
        'str': 'int16',
        'per_plio': 8
    }
}

# ---------------- Common helpers ----------------

def tile_matrix(matrix, row_tiles, col_tiles):  # (R,C) -> (R/r, C/c, r, c).flatten()
    rows, cols = matrix.shape
    assert rows % row_tiles == 0 and cols % col_tiles == 0, "Matrix must be divisible by block sizes"
    reshaped = matrix.reshape(rows // row_tiles, row_tiles, cols // col_tiles, col_tiles)
    transposed = reshaped.transpose(0, 2, 1, 3)  # (R/r, C/c, r, c)
    return transposed.flatten()

# ---------------- Layer classes ----------------

class Layer:
    def emit(self, idx, x_in, y_ref, iterations):
        raise NotImplementedError
    def forward(self, x_in):
        raise NotImplementedError

class Dense(Layer):
    """
    Dense operating row-wise on (R, K) -> (R, N); AIE tiles m=2, k=8, n=8.
    Accepts either 2-D (R,K) or 3-D NHWC (H,W,C). For 3-D, it flattens to (H*W, C).
    """
    def __init__(self, N, shift=0, relu=False, m_tile=2, k_tile=8, n_tile=8, dtype='i8'):
        self.N = N
        self.shift = shift
        self.relu = relu
        self.m_tile = m_tile
        self.k_tile = k_tile
        self.n_tile = n_tile
        self._last_in2d = None  # cached 2D view used in emit
        self.dtype = dtype

    def _as_2d(self, x_in: np.ndarray):
        if x_in.ndim == 2:
            return x_in
        elif x_in.ndim == 3:
            XH, XW, XC = x_in.shape
            return x_in.reshape(XH * XW, XC)
        else:
            raise ValueError(f"Dense expects 2D or 3D input, got shape {x_in.shape}")

    def forward(self, x_in):
        x2d = self._as_2d(x_in)
        self._last_in2d = x2d  # remember for emit()
        R, K = x2d.shape
        assert (R % self.m_tile) == 0 and (K % self.k_tile) == 0 and (self.N % self.n_tile) == 0
        self.K = K
        # random weights (K,N)
        self.W = np.random.randint(0, 128, size=(K, self.N), dtype=TY_DICT[self.dtype]['np'])
        # reference
        y = (x2d.astype(np.int32) @ self.W.astype(np.int32))
        y = (y >> self.shift).astype(TY_DICT[self.dtype]['np'])
        if self.relu:
            y = np.maximum(0, y)
        return y

    def emit(self, idx, x_in, y_ref, iterations):
        m, k, n = self.m_tile, self.k_tile, self.n_tile

        # Use the same 2D view we used during forward
        x2d = self._last_in2d if self._last_in2d is not None else self._as_2d(x_in)

        # weights tiled KxN -> k_tiled
        k_tiled = tile_matrix(self.W, k, n)

        # IO files (dense uses tiled dumps)
        x_tiled = tile_matrix(x2d, m, k)
        a_tiled = tile_matrix(y_ref, m, n)
        np.savetxt(f"data/x{idx}.txt", np.tile(x_tiled, (iterations, 1)).reshape(-1, TY_DICT[self.dtype]['per_plio']), fmt="%s", delimiter=" ")
        np.savetxt(f"data/a{idx}.txt", np.tile(a_tiled, (iterations, 1)).reshape(-1, TY_DICT[self.dtype]['per_plio']), fmt="%s", delimiter=" ")

        ty_str = TY_DICT[self.dtype]['str']

        with open(f"model/layer_{idx}.cc", "a") as f:
            f.write(f'''
#define DTYPE {ty_str}
#define mm_m {m}
#define mm_k {k}
#define mm_n {n}
#define mm_M {x2d.shape[0]}
#define mm_K {x2d.shape[1]}
#define mm_N {self.W.shape[1]}
#define SHIFT {self.shift}
#define DO_RELU {str(self.relu).lower()}

#include <cstdint>
__attribute__((section(".data"))) alignas(32) {ty_str}_t matB [{k_tiled.size}] = {{ {", ".join(str(int(x)) for x in k_tiled)} }};

#include "dense_window.h"

void f{idx}(input_window_{ty_str} * __restrict in, output_window_{ty_str} * __restrict out){{ dense(in, out);}}
''')

        # Connect bytes from the *previous* layer as-is (window size is just count of dtype)
        in_port   = "AIE_IN" if idx == 0 else f"layers[{idx-1}]"
        num_bytes = x_in.size * x_in.itemsize
        with open("model/layer_graph.h", "a") as f:
            f.write(f"layers[{idx}] = kernel::create(f{idx});\n")
            f.write(f'source(layers[{idx}]) = "layer_{idx}.cc";\n')
            f.write(f"connect<window<{num_bytes}>>({in_port}.out[0], layers[{idx}].in[0]);\n\n")
            if idx == 0 and num_bytes > 32768:
                f.write(f"single_buffer(layers[{idx}].in[0]);\n")


# ---------------- Sequential-ish model ----------------

class Sequential:
    def __init__(self, iterations=1, dtype='i8'):
        self.layers = []
        self.iterations = iterations
        self.dtype = dtype

    def add(self, layer: Layer):
        self.layers.append(layer)

    def build_and_emit(self, x0: np.ndarray):
        """
        Runs reference forward pass, emits per-layer .cc and layer_graph wiring,
        and returns final reference output.
        """
        # ensure clean dirs already set up by caller; we only write files here
        # also create empty layer_graph for appends
        open("model/layer_graph.h", "w").close()

        x = x0
        for idx, layer in enumerate(self.layers):
            y = layer.forward(x)
            layer.emit(idx, x, y, self.iterations)
            x = y

        N_LAYERS = len(self.layers)
        in_bytes = x0.size * x0.itemsize

        # last layer output file for final compare (depends on layer type)
        last = self.layers[-1]
        if isinstance(last, Dense):
            # dense writes in tiled (m=2, n=8) layout at the output edge
            m_tile, n_tile = last.m_tile, last.n_tile
            tiled_last = tile_matrix(x, m_tile, n_tile)
            np.savetxt("data/out_ref.txt",
                       np.tile(tiled_last, (self.iterations,1)).reshape(-1,TY_DICT[self.dtype]['per_plio']),
                       fmt="%s", delimiter=" ")
            out_bytes = x.size * x.itemsize

        with open("model/layer_graph.h", "a") as f:
            if out_bytes >= 32768:
                f.write(f"single_buffer(layers[{N_LAYERS-1}].out[0]);\n")
            f.write(f"connect<window<{out_bytes}>>(layers[{N_LAYERS-1}].out[0], AIE_OUT.in[0]);\n")
        
        ty_str = TY_DICT[self.dtype]['str']
        # finalize include.h
        with open("model/include.h", "w") as f:
            f.write(f'#define DTYPE {ty_str}\n')
            f.write(f'#define N_LAYERS {N_LAYERS}\n')
            f.write(f'#define ITERATIONS {self.iterations}\n')
            f.write(f'#define TOT_OUT_BYTES {out_bytes*self.iterations}\n')
            f.write(f'#define TOT_IN_BYTES {in_bytes*self.iterations}\n')
            for idx in range(N_LAYERS):
                f.write(f'void f{idx}(input_window_{ty_str} * __restrict, output_window_{ty_str} * __restrict);\n')

        return x


# ---------------- Main ----------------

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Run AIE dense kernel sim & check.")
    ap.add_argument("--dtype",   type=str, default="i8", help="dtype: i8 or i16 (default: i8)")
    ap.add_argument("--batch",   "-b", type=int, default=4, help="Batch size (default: 4)")
    ap.add_argument("--inputs",  "-i", type=int, default=128, help="Number of inputs/features (default: 128)")
    ap.add_argument("--outputs", "-o", type=int, default=128, help="Number of outputs (default: 128)")
    args = ap.parse_args()
    BATCH, INPUTS, OUTPUTS, dtype = args.batch, args.inputs, args.outputs, args.dtype

    if dtype == 'i8':
        m_tile, k_tile, n_tile = 2, 8, 8
    elif dtype == 'i16':
        m_tile, k_tile, n_tile = 2, 4, 8

    iterations = 11

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
    model = Sequential(iterations=iterations, dtype=dtype)
    model.add(Dense(N=OUTPUTS, shift=5, relu=True, m_tile=m_tile, k_tile=k_tile, n_tile=n_tile, dtype=dtype))

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
        print(f"\n\n Success: Outputs match ({out_sim.shape})\n\n{out_sim}\n\n")
    else:
        print("\n\nError: Output does not match\n")
        print(f"Simulation Output ({out_sim.shape}):\n{out_sim}\n")
        print(f"Expected output ({out_ref.shape}):\n{out_ref}\n")