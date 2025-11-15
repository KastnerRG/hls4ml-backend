import numpy as np

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

def tile_matrix(matrix, row_tiles, col_tiles):  # (R,C) -> (R/r, C/c, r, c).flatten()
    rows, cols = matrix.shape
    assert rows % row_tiles == 0 and cols % col_tiles == 0, "Matrix must be divisible by block sizes"
    reshaped = matrix.reshape(rows // row_tiles, row_tiles, cols // col_tiles, col_tiles)
    transposed = reshaped.transpose(0, 2, 1, 3)  # (R/r, C/c, r, c)
    return transposed.flatten()

class Layer:
    def __init__(self):
        self._decls = []
    def emit(self, idx, x_in, y_ref, iterations, layers):
        raise NotImplementedError
    def forward(self, x_in):
        raise NotImplementedError

class Dense(Layer):
    """
    Dense operating row-wise on (R, K) -> (R, N); AIE tiles m=2, k=8, n=8.
    Accepts either 2-D (R,K) or 3-D NHWC (H,W,C). For 3-D, it flattens to (H*W, C).
    """
    def __init__(self, N, shift=0, relu=False, m_tile=2, k_tile=8, n_tile=8, dtype='i16', dataflow='stream', free=False, input_plios=1, **kwargs):
        super().__init__()
        self.N = N
        self.shift = shift
        self.relu = relu
        self.m_tile = m_tile
        self.k_tile = k_tile
        self.n_tile = n_tile
        self._last_in2d = None  # cached 2D view used in emit
        self.dtype = dtype
        self.dataflow = dataflow
        self.free = free
        self.input_plios = input_plios

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

    def emit(self, idx, x_in, y_ref, iterations, layers):
        m, k, n = self.m_tile, self.k_tile, self.n_tile

        # Use the same 2D view we used during forward
        x2d = self._last_in2d if self._last_in2d is not None else self._as_2d(x_in)
        Tm = x2d.shape[0] // m
        Tn = self.W.shape[1] // n

        # IO files (dense uses tiled dumps)
        per_plio = TY_DICT[self.dtype]['per_plio']
        x_tiled = tile_matrix(x2d, m, k)
        k_tiled_full = tile_matrix(self.W, k, n)
        a_tiled = tile_matrix(y_ref, m, n)
        np.savetxt(f"data/a{idx}.txt", np.tile(a_tiled, (iterations, 1)).reshape(-1, per_plio), fmt="%s", delimiter=" ")

        ty_str = TY_DICT[self.dtype]['str']

        if self.input_plios == 1:
            np.savetxt(f"data/x{idx}.txt", np.tile(x_tiled, (iterations, 1)).reshape(-1, per_plio), fmt="%s", delimiter=" ")

            with open(f"model/layer_{idx}.cc", "a") as f:
                if self.free:
                    f.write(f'#define FREE\n')
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
__attribute__((section(".data"))) alignas(32) {ty_str}_t matB [{k_tiled_full.size}] = {{ {", ".join(str(int(x)) for x in k_tiled_full)} }};

#include "dense_{self.dataflow}.h"

void f{idx}(input_{self.dataflow}_{ty_str} * __restrict in, output_{self.dataflow}_{ty_str} * __restrict out){{ dense(in, out);}}
''')
            self._decls = [f'void f{idx}(input_{self.dataflow}_{ty_str} * __restrict, output_{self.dataflow}_{ty_str} * __restrict);']
        else:
            assert self.dataflow == 'stream', "Multiple input PLIOs currently require stream dataflow"
            assert idx == 0, "Multiple input PLIOs supported only on the first layer for now"
            Tk_total = x2d.shape[1] // k
            assert (Tk_total % self.input_plios) == 0, "Tk must divide across input PLIOs"
            assert (self.K % self.input_plios) == 0, "K must divide across input PLIOs"
            Tk_per = Tk_total // self.input_plios
            K_per = self.K // self.input_plios
            vecs_per_tile = Tm * Tn
            vec_elems = m * n
            partial_elems = vecs_per_tile * vec_elems
            partial_bytes = partial_elems * 4  # int32 partial sums
            xt = x2d.reshape(Tm, m, Tk_total, k).transpose(0, 2, 1, 3)
            wt = self.W.reshape(self.K // k, k, self.W.shape[1] // n, n).transpose(0, 2, 1, 3)
            recon_xt = np.empty_like(xt)
            recon_wt = np.empty_like(wt)
            chunks_x = []
            chunks_k = []

            for tile_idx in range(self.input_plios):
                x_chunk = xt[:, tile_idx*Tk_per:(tile_idx+1)*Tk_per, :, :]
                chunk_flat = x_chunk.flatten()
                chunks_x.append(chunk_flat.copy())
                recon_xt[:, tile_idx*Tk_per:(tile_idx+1)*Tk_per, :, :] = chunk_flat.reshape(Tm, Tk_per, m, k)
                np.savetxt(f"data/x{idx}_{tile_idx}.txt",
                           np.tile(chunk_flat, (iterations, 1)).reshape(-1, per_plio),
                           fmt="%s", delimiter=" ", newline="\n")

                W_chunk = self.W[tile_idx*K_per:(tile_idx+1)*K_per, :]
                k_tiled_chunk = tile_matrix(W_chunk, k, n)
                chunks_k.append(k_tiled_chunk.copy())
                recon_wt[tile_idx*(K_per//k):(tile_idx+1)*(K_per//k), :, :, :] = k_tiled_chunk.reshape(K_per//k, wt.shape[1], k, n)

                with open(f"model/layer_{idx}_partial{tile_idx}.cc", "w") as f:
                    if self.free:
                        f.write(f'#define FREE\n')
                    f.write(f'''
#define DTYPE {ty_str}
#define mm_m {m}
#define mm_k {k}
#define mm_n {n}
#define mm_M {x2d.shape[0]}
#define mm_K {K_per}
#define mm_N {self.W.shape[1]}
#define SHIFT 0
#define DO_RELU false
#define DENSE_PARTIAL_ACC 1

#include <cstdint>
__attribute__((section(".data"))) alignas(32) {ty_str}_t matB [{k_tiled_chunk.size}] = {{ {", ".join(str(int(x)) for x in k_tiled_chunk)} }};

#include "dense_{self.dataflow}.h"

void f{idx}_partial{tile_idx}(input_{self.dataflow}_{ty_str} * __restrict in, adf::output_buffer<int32_t> & __restrict out){{ dense(in, out);}}
''')
                if tile_idx == 0:
                    self._decls = []
                self._decls.append(f'void f{idx}_partial{tile_idx}(input_{self.dataflow}_{ty_str} * __restrict, adf::output_buffer<int32_t> & __restrict);')
            np.testing.assert_array_equal(recon_xt.flatten(), x_tiled)
            np.testing.assert_array_equal(recon_wt.flatten(), k_tiled_full)

            agg_inputs = ", ".join([f"adf::input_buffer<int32_t> & __restrict in{p}" for p in range(self.input_plios)])
            iter_decls = "\n  ".join([f"auto it{p} = aie::begin_vector<mm_m * mm_n>(in{p});" for p in range(self.input_plios)])
            accum_lines = "\n    ".join([f"acc_vec = aie::add(acc_vec, *it{p}++);" for p in range(1, self.input_plios)])
            accum_block = f"\n    {accum_lines}" if accum_lines else ""
            call_inputs = ", ".join([f"in{p}" for p in range(self.input_plios)])
            with open(f"model/layer_{idx}.cc", "w") as f:
                if self.free:
                    f.write(f'#define FREE\n')
                f.write(f'''
#define DTYPE {ty_str}
#define mm_m {m}
#define mm_k {k}
#define mm_n {n}
#define mm_M {x2d.shape[0]}
#define mm_K {self.K}
#define mm_N {self.W.shape[1]}
#define SHIFT {self.shift}
#define DO_RELU {str(self.relu).lower()}
#define NUM_PARTIALS {self.input_plios}
#ifndef NB
#define NB 4
#endif

#include <adf.h>
#include "aie_api/aie.hpp"
#include <algorithm>
#include <limits>

static inline void dense_reduce({agg_inputs}, output_{self.dataflow}_{ty_str} * __restrict out){{
  constexpr unsigned VEC = mm_m * mm_n;
  const unsigned total_vec = (mm_M / mm_m) * (mm_N / mm_n);
  alignas(32) int32_t acc[VEC];
  alignas(32) DTYPE out_buf[VEC];
  {iter_decls}
  for (unsigned vec = 0; vec < total_vec; ++vec)
  chess_prepare_for_pipelining
  {{
    aie::vector<int32_t, VEC> acc_vec = *it0++;{accum_block}
    aie::store_v(acc, acc_vec);
    for (unsigned lane = 0; lane < VEC; ++lane)
    {{
      int32_t val = acc[lane];
      if (SHIFT > 0)
      {{
        val >>= SHIFT;
      }}
      DTYPE q = static_cast<DTYPE>(val);
      if (DO_RELU && q < 0)
      {{
        q = 0;
      }}
      out_buf[lane] = q;
    }}
    aie::vector<DTYPE, VEC> v = aie::load_v<VEC>(out_buf);
    writeincr(out, v);
  }}
}}

void f{idx}({agg_inputs}, output_{self.dataflow}_{ty_str} * __restrict out){{ dense_reduce({call_inputs}, out); }}
''')
            agg_proto = f'void f{idx}({agg_inputs}, output_{self.dataflow}_{ty_str} * __restrict);'
            self._decls.append(agg_proto)

        # Connect bytes from the *previous* layer as-is (window size is just count of dtype)
        in_port   = "AIE_IN[0]" if idx == 0 else f"layers[{idx-1}]"
        num_bytes = x_in.size * x_in.itemsize
        with open("model/layer_graph.h", "a") as f:
            if self.input_plios == 1:
                f.write(f"layers[{idx}] = kernel::create(f{idx});\n")
                f.write(f'source(layers[{idx}]) = "layer_{idx}.cc";\n')
                if self.dataflow == 'stream':
                    f.write(f"auto c{idx} = connect<stream>({in_port}.out[0], layers[{idx}].in[0]);\n")
                elif self.dataflow == 'window':
                    f.write(f"connect<window<{num_bytes}>>({in_port}.out[0], layers[{idx}].in[0]);\n\n")
                if idx == 0 and num_bytes > 32768:
                    f.write(f"single_buffer(layers[{idx}].in[0]);\n")
            else:
                for tile_idx in range(self.input_plios):
                    f.write(f"kernel layer_{idx}_partial{tile_idx} = kernel::create(f{idx}_partial{tile_idx});\n")
                    f.write(f'source(layer_{idx}_partial{tile_idx}) = "layer_{idx}_partial{tile_idx}.cc";\n')
                    f.write(f"runtime<ratio>(layer_{idx}_partial{tile_idx}) = 1.0;\n")
                    f.write(f"connect<stream>(AIE_IN[{tile_idx}].out[0], layer_{idx}_partial{tile_idx}.in[0]);\n")
                    f.write(f"dimensions(layer_{idx}_partial{tile_idx}.out[0]) = {{ {partial_elems} }};\n")
                f.write(f"layers[{idx}] = kernel::create(f{idx});\n")
                f.write(f'source(layers[{idx}]) = "layer_{idx}.cc";\n')
                f.write(f"runtime<ratio>(layers[{idx}]) = 1.0;\n")
                for tile_idx in range(self.input_plios):
                    f.write(f"dimensions(layers[{idx}].in[{tile_idx}]) = {{ {partial_elems} }};\n")
                    f.write(f"connect<>(layer_{idx}_partial{tile_idx}.out[0], layers[{idx}].in[{tile_idx}]);\n")


class Sequential:
    def __init__(self, iterations=1, dtype='i16', dataflow='stream', free=False, input_plios=1, **kwargs):
        self.layers = []
        self.iterations = iterations
        self.dtype = dtype
        self.dataflow = dataflow
        self.free = free
        self.input_plios = input_plios
        self.output_plios = 1

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
            layer.emit(idx, x, y, self.iterations, self.layers)
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

            if self.dataflow == 'stream':
                f.write(f"auto c{N_LAYERS} = connect<stream>(layers[{N_LAYERS-1}].out[0], AIE_OUT[0].in[0]);\n")
                # f.write(f"fifo_depth(c{N_LAYERS}) = 32;\n")
            elif self.dataflow == 'window':
                f.write(f"connect<window<{out_bytes}>>(layers[{N_LAYERS-1}].out[0], AIE_OUT[0].in[0]);\n")

        ty_str = TY_DICT[self.dtype]['str']

        # finalize include.h
        with open("model/include.h", "w") as f:
            if self.free:
                f.write(f'#define FREE\n')
            f.write(f'#define DTYPE {ty_str}\n')
            f.write(f'#define N_LAYERS {N_LAYERS}\n')
            f.write(f'#define ITERATIONS {self.iterations}\n')
            f.write(f'#define TOT_OUT_BYTES {out_bytes*self.iterations}\n')
            f.write(f'#define TOT_IN_BYTES {in_bytes*self.iterations}\n')
            f.write(f'#define NUM_INPUT_PLIO {self.input_plios}\n')
            f.write(f'#define NUM_OUTPUT_PLIO {self.output_plios}\n')
            for idx, layer in enumerate(self.layers):
                decls = getattr(layer, "_decls", [])
                if decls:
                    for decl in decls:
                        f.write(f'{decl}\n')
                else:
                    f.write(f'void f{idx}(input_{self.dataflow}_{ty_str} * __restrict, output_{self.dataflow}_{ty_str} * __restrict);\n')
        return x
