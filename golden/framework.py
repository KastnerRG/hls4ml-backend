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
    if rows % row_tiles != 0 or cols % col_tiles != 0:
        raise AssertionError(f"Matrix must be divisible by block sizes: rows={rows}, row_tile={row_tiles}, cols={cols}, col_tile={col_tiles}")
    reshaped = matrix.reshape(rows // row_tiles, row_tiles, cols // col_tiles, col_tiles)
    transposed = reshaped.transpose(0, 2, 1, 3)  # (R/r, C/c, r, c)
    return transposed.flatten()

class Layer:
    def __init__(self):
        self._decls = []
        self.input_mode = 'stream'
        self.output_mode = 'stream'
    def emit(self, idx, x_in, y_ref, iterations, layers):
        raise NotImplementedError
    def forward(self, x_in):
        raise NotImplementedError

class Dense(Layer):
    """
    Dense operating row-wise on (R, K) -> (R, N); AIE tiles m=2, k=8, n=8.
    Accepts either 2-D (R,K) or 3-D NHWC (H,W,C). For 3-D, it flattens to (H*W, C).
    """
    def __init__(self, N, shift=0, relu=False, m_tile=2, k_tile=8, n_tile=8, dtype='i16', dataflow='stream', free=False, input_plios=1, output_plios=1, cascade_out=False, cascade_in=False, input_shift=0, input_relu=False, input_m_tile=None, input_n_tile=None, **kwargs):
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
        self.output_plios = output_plios
        self.cascade_out = cascade_out
        if self.cascade_out:
            self.output_mode = 'cascade'
        self.cascade_in = cascade_in
        if self.cascade_in:
            self.input_mode = 'cascade'
        self.input_shift = input_shift
        self.input_relu = input_relu
        self.input_m_tile = input_m_tile if input_m_tile is not None else m_tile
        self.input_n_tile = input_n_tile if input_n_tile is not None else k_tile

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
    def get_output_layout(self):
        return (self.m_tile, self.n_tile)

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
        stream_type = f"output_{self.dataflow}_{ty_str}"
        cascade_tag = {'i8': 'acc32', 'i16': 'acc48'}[self.dtype]

        cascin_kernel = None
        cascin_func = None
        cascin_decl = None
        split_stage_names = []
        split_stage_funcs = []
        split_stage_decls = []
        if self.input_mode == 'cascade':
            assert idx > 0, "Cascade input requires previous layer"
            prev_layer = layers[idx-1]
            assert getattr(prev_layer, "output_mode", "stream") == "cascade", "Previous layer must output cascade"
            if self.input_plios == 1:
                cascin_kernel = f"layer_{idx}_cascin"
                cascin_func = f"f{idx}_cascin"
                with open(f"model/{cascin_kernel}.cc", "w") as f:
                    if self.free:
                        f.write(f'#define FREE\n')
                    f.write(f"#define DENSE_CASC_TYPE {cascade_tag}\n")
                    f.write(f'''\
#define DTYPE {ty_str}
#define mm_m {self.input_m_tile}
#define mm_k {self.k_tile}
#define mm_n {self.input_n_tile}
#define mm_M {x2d.shape[0]}
#define mm_K {x2d.shape[1]}
#define mm_N {x2d.shape[1]}
#define SHIFT {self.input_shift}
#define DO_RELU {str(self.input_relu).lower()}

#include <cstdint>
__attribute__((section(".data"))) alignas(32) {ty_str}_t matB[1] = {{ 0 }};

#include "dense_{self.dataflow}.h"

void {cascin_func}(input_cascade<{cascade_tag}> * __restrict casc_in, output_{self.dataflow}_{ty_str} * __restrict out){{ dense_quant(casc_in, out); }}
''')
                cascin_decl = f"void {cascin_func}(input_cascade<{cascade_tag}> * __restrict, output_{self.dataflow}_{ty_str} * __restrict);"
            else:
                Tk_total = x2d.shape[1] // k
                Tk_per = Tk_total // self.input_plios
                assert Tk_total % self.input_plios == 0, "Cascade split Tk mismatch"
                for split_idx in range(self.input_plios):
                    stage_kernel = f"layer_{idx}_split{split_idx}"
                    stage_func = f"f{idx}_split{split_idx}"
                    has_casc_out = split_idx != self.input_plios - 1
                    Tk_residual = Tk_total - split_idx * Tk_per
                    with open(f"model/{stage_kernel}.cc", "w") as f:
                        if self.free:
                            f.write(f'#define FREE\n')
                        f.write(f"#define DENSE_CASC_TYPE {cascade_tag}\n")
                        f.write(f"#define SPLIT_TK {Tk_per}\n")
                        f.write(f"#define TK_RESIDUAL {Tk_residual}\n")
                        f.write(f"#define HAS_CASC_OUT {1 if has_casc_out else 0}\n")
                        f.write(f'''\
#define DTYPE {ty_str}
#define mm_m {self.input_m_tile}
#define mm_k {self.k_tile}
#define mm_n {self.input_n_tile}
#define mm_M {x2d.shape[0]}
#define mm_K {(Tk_residual) * self.k_tile}
#define mm_N {x2d.shape[1]}
#define SHIFT {self.input_shift}
#define DO_RELU {str(self.input_relu).lower()}

#include <cstdint>
__attribute__((section(".data"))) alignas(32) {ty_str}_t matB[1] = {{ 0 }};

#include "dense_cascade_split_stage.h"

''')
                        args = [f"input_cascade<{cascade_tag}> * __restrict casc_in"]
                        if has_casc_out:
                            args.append(f"output_cascade<{cascade_tag}> * __restrict casc_out")
                        args.append(f"output_{self.dataflow}_{ty_str} * __restrict out")
                        f.write(f"void {stage_func}({', '.join(args)}){{ dense_cascade_split_stage(casc_in{', casc_out' if has_casc_out else ''}, out); }}\n")
                    split_stage_names.append(stage_kernel)
                    split_stage_funcs.append((stage_func, has_casc_out))
                    decl_args = ", ".join(args)
                    split_stage_decls.append(f"void {stage_func}({decl_args});")

        if self.input_plios == 1:
            if self.input_mode == 'stream':
                np.savetxt(f"data/x{idx}.txt", np.tile(x_tiled, (iterations, 1)).reshape(-1, per_plio), fmt="%s", delimiter=" ")

            if self.output_plios == 1:
                with open(f"model/layer_{idx}.cc", "a") as f:
                    if self.free:
                        f.write(f'#define FREE\n')
                    if self.cascade_out:
                        f.write(f"#define DENSE_CASC_TYPE {cascade_tag}\n")
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

void f{idx}(input_{self.dataflow}_{ty_str} * __restrict in, { 'output_cascade<' + cascade_tag + '> * __restrict casc_out' if self.cascade_out else stream_type + ' * __restrict out' }){{ {"dense_first(in, casc_out);" if self.cascade_out else "dense_single(in, out);"} }}
''')
                self._decls = self._decls if self._decls else []
                if self.cascade_out:
                    self._decls.append(f'void f{idx}(input_{self.dataflow}_{ty_str} * __restrict, output_cascade<{cascade_tag}> * __restrict);')
                else:
                    self._decls.append(f'void f{idx}(input_{self.dataflow}_{ty_str} * __restrict, {stream_type} * __restrict);')
                if cascin_decl:
                    self._decls.append(cascin_decl)
            else:
                assert self.W.shape[1] % self.output_plios == 0, "Output count must divide over output PLIOs"
                assert idx == len(layers) - 1, "Output PLIO tiling currently supported only on the final layer"
                n_per = self.W.shape[1] // self.output_plios
                stream_type = f"output_{self.dataflow}_{ty_str}"
                kernel_handles = []
                self._decls = []
                self._multi_output_handles = []

                Tk_total = x2d.shape[1] // k
                for tile_idx in range(self.output_plios):
                    start = tile_idx * n_per
                    end = (tile_idx + 1) * n_per
                    W_chunk = self.W[:, start:end]
                    chunk_tiled = tile_matrix(W_chunk, k, n).reshape(Tk_total, n_per // n, k, n)
                    nb_vectors = min(2, chunk_tiled.shape[1])
                    blocksN = (chunk_tiled.shape[1] + nb_vectors - 1) // nb_vectors
                    packed = np.zeros((Tk_total, blocksN, nb_vectors, k, n), dtype=chunk_tiled.dtype)
                    for blk in range(blocksN):
                        for slot in range(nb_vectors):
                            vec_idx = blk * nb_vectors + slot
                            if vec_idx < chunk_tiled.shape[1]:
                                packed[:, blk, slot, :, :] = chunk_tiled[:, vec_idx, :, :]
                    chunk_flat = packed.flatten()
                    assert chunk_flat.size == Tk_total * blocksN * nb_vectors * k * n
                    handle = f"layer_{idx}_out{tile_idx}"
                    func = f"f{idx}_out{tile_idx}"

                    with open(f"model/{handle}.cc", "w") as f:
                        if self.free:
                            f.write(f'#define FREE\n')
                        f.write(f"#define DENSE_IN_CASC_TYPE {cascade_tag}\n")
                        f.write(f'''\
#define DTYPE {ty_str}
#define mm_m {m}
#define mm_k {k}
#define mm_n {n}
#define mm_M {x2d.shape[0]}
#define mm_K {x2d.shape[1]}
#define mm_N {n_per}
#define SHIFT {self.shift}
#define DO_RELU {str(self.relu).lower()}

#include <cstdint>
__attribute__((section(".data"))) alignas(32) {ty_str}_t matB [{chunk_flat.size}] = {{ {", ".join(str(int(x)) for x in chunk_flat)} }};

#include "dense_stream_out.h"

''')
                        if tile_idx == 0:
                            f.write(f"void {func}(input_{self.dataflow}_{ty_str} * __restrict in, output_cascade<{cascade_tag}> * __restrict casc_out, {stream_type} * __restrict out){{ dense_out_first(in, casc_out, out); }}\n")
                        elif tile_idx == self.output_plios - 1:
                            f.write(f"void {func}(input_cascade<{cascade_tag}> * __restrict casc_in, {stream_type} * __restrict out){{ dense_out_last(casc_in, out); }}\n")
                        else:
                            f.write(f"void {func}(input_cascade<{cascade_tag}> * __restrict casc_in, output_cascade<{cascade_tag}> * __restrict casc_out, {stream_type} * __restrict out){{ dense_out_middle(casc_in, casc_out, out); }}\n")

                    kernel_handles.append(handle)
                    proto_parts = []
                    if tile_idx == 0:
                        proto_parts.append(f"input_{self.dataflow}_{ty_str} * __restrict")
                    else:
                        proto_parts.append(f"input_cascade<{cascade_tag}> * __restrict")
                    if tile_idx != self.output_plios - 1:
                        proto_parts.append(f"output_cascade<{cascade_tag}> * __restrict")
                    proto_parts.append(f"{stream_type} * __restrict")
                    self._decls.append(f"void {func}({', '.join(proto_parts)});")

                    stream_port_idx = 1 if tile_idx != self.output_plios - 1 else 0
                    self._multi_output_handles.append((handle, stream_port_idx))

                with open("model/layer_graph.h", "a") as f:
                    for tile_idx, handle in enumerate(kernel_handles):
                        func = f"f{idx}_out{tile_idx}"
                        f.write(f"kernel {handle} = kernel::create({func});\n")
                        f.write(f'source({handle}) = "{handle}.cc";\n')
                        f.write(f"runtime<ratio>({handle}) = 1.0;\n")
                    f.write(f"layers[{idx}] = {kernel_handles[-1]};\n")
                    in_port = "AIE_IN[0]" if idx == 0 else f"layers[{idx-1}]"
                    f.write(f"connect<stream>({in_port}.out[0], {kernel_handles[0]}.in[0]);\n")
                    for tile_idx in range(1, self.output_plios):
                        prev = kernel_handles[tile_idx-1]
                        curr = kernel_handles[tile_idx]
                        f.write(f"connect<cascade>({prev}.out[0], {curr}.in[0]);\n")
                return

        if self.input_plios > 1:
            assert self.dataflow == 'stream', "Multiple input PLIOs currently require stream dataflow"
            if self.input_mode == 'stream':
                assert idx == 0, "Multiple input PLIOs supported only on the first layer for now"
            Tk_total = x2d.shape[1] // k
            assert (Tk_total % self.input_plios) == 0, "Tk must divide across input PLIOs"
            assert (self.K % self.input_plios) == 0, "K must divide across input PLIOs"
            Tk_per = Tk_total // self.input_plios
            K_per = self.K // self.input_plios
            vecs_per_tile = Tm * Tn
            vec_elems = m * n
            xt = x2d.reshape(x2d.shape[0] // m, m, Tk_total, k).transpose(0, 2, 1, 3)
            wt = self.W.reshape(self.K // k, k, self.W.shape[1] // n, n).transpose(0, 2, 1, 3)
            recon_xt = np.empty_like(xt)
            recon_wt = np.empty_like(wt)
            stream_type = f"output_{self.dataflow}_{ty_str}"

            for tile_idx in range(self.input_plios):
                is_last = tile_idx == self.input_plios - 1
                has_casc_in = tile_idx != 0
                if is_last and self.input_plios > 1:
                    has_casc_out = True
                    has_stream_out = False
                    shift_here = 0
                    relu_here = False
                else:
                    has_casc_out = not is_last
                    has_stream_out = is_last
                    shift_here = self.shift if is_last else 0
                    relu_here = self.relu if is_last else False

                x_chunk = xt[:, tile_idx*Tk_per:(tile_idx+1)*Tk_per, :, :]
                chunk_flat = x_chunk.flatten()
                recon_xt[:, tile_idx*Tk_per:(tile_idx+1)*Tk_per, :, :] = chunk_flat.reshape(Tm, Tk_per, m, k)
                if self.input_mode == 'stream':
                    np.savetxt(f"data/x{idx}_{tile_idx}.txt",
                               np.tile(chunk_flat, (iterations, 1)).reshape(-1, per_plio),
                               fmt="%s", delimiter=" ", newline="\n")

                W_chunk = self.W[tile_idx*K_per:(tile_idx+1)*K_per, :]
                k_tiles = tile_matrix(W_chunk, k, n).reshape(K_per//k, wt.shape[1], k, n)
                recon_wt[tile_idx*(K_per//k):(tile_idx+1)*(K_per//k), :, :, :] = k_tiles

                nb_vectors = min(2, k_tiles.shape[1])
                blocksN = (k_tiles.shape[1] + nb_vectors - 1) // nb_vectors
                packed = np.zeros((K_per//k, blocksN, nb_vectors, k, n), dtype=k_tiles.dtype)
                for blk in range(blocksN):
                    for slot in range(nb_vectors):
                        vec_idx = blk * nb_vectors + slot
                        if vec_idx < k_tiles.shape[1]:
                            packed[:, blk, slot, :, :] = k_tiles[:, vec_idx, :, :]
                packed_flat = packed.flatten()

                target = f"model/layer_{idx}.cc" if (is_last and self.input_plios == 1) else f"model/layer_{idx}_partial{tile_idx}.cc"
                func_name = f"f{idx}" if (is_last and self.input_plios == 1) else f"f{idx}_partial{tile_idx}"

                with open(target, "w") as f:
                    if self.free:
                        f.write(f'#define FREE\n')
                    f.write(f"#define DENSE_CASC_TYPE {cascade_tag}\n")
                    f.write(f'''\
#define DTYPE {ty_str}
#define mm_m {m}
#define mm_k {k}
#define mm_n {n}
#define mm_M {x2d.shape[0]}
#define mm_K {K_per}
#define mm_N {self.W.shape[1]}
#define SHIFT {shift_here}
#define DO_RELU {str(relu_here).lower()}

#include <cstdint>
__attribute__((section(".data"))) alignas(32) {ty_str}_t matB [{packed_flat.size}] = {{ {", ".join(str(int(x)) for x in packed_flat)} }};

#include "dense_{self.dataflow}.h"

void {func_name}(input_{self.dataflow}_{ty_str} * __restrict in{', input_cascade<' + cascade_tag + '> * __restrict casc_in' if has_casc_in else ''}{', output_cascade<' + cascade_tag + '> * __restrict casc_out' if has_casc_out else ''}{', ' + stream_type + ' * __restrict out' if has_stream_out else ''}){{ {"dense_single(in, out);" if self.input_plios == 1 else ("dense_first(in, casc_out);" if tile_idx == 0 else ("dense_last_casc(in, casc_in, casc_out);" if (is_last and self.input_plios > 1) else ("dense_last(in, casc_in, out);" if is_last else "dense_middle(in, casc_in, casc_out);")))} }}
''')
                if tile_idx == 0:
                    self._decls = []
                proto_parts = [f"input_{self.dataflow}_{ty_str} * __restrict"]
                if has_casc_in:
                    proto_parts.append(f"input_cascade<{cascade_tag}> * __restrict")
                if has_casc_out:
                    proto_parts.append(f"output_cascade<{cascade_tag}> * __restrict")
                if has_stream_out:
                    proto_parts.append(f"{stream_type} * __restrict")
                self._decls.append(f"void {func_name}({', '.join(proto_parts)});")

                if self.input_plios > 1 and not self.cascade_out:
                    quant_target = f"model/layer_{idx}_quant.cc"
                    with open(quant_target, "w") as f:
                        if self.free:
                            f.write(f'#define FREE\n')
                        f.write(f"#define DENSE_CASC_TYPE {cascade_tag}\n")
                        f.write(f'''\
#define DTYPE {ty_str}
#define mm_m {m}
#define mm_k {k}
#define mm_n {n}
#define mm_M {x2d.shape[0]}
#define mm_K {K_per}
#define mm_N {self.W.shape[1]}
#define SHIFT {self.shift}
#define DO_RELU {str(self.relu).lower()}

#include <cstdint>
__attribute__((section(".data"))) alignas(32) {ty_str}_t matB[1] = {{ 0 }};

#include "dense_{self.dataflow}.h"

void f{idx}_quant(input_cascade<{cascade_tag}> * __restrict casc_in, {stream_type} * __restrict out){{ dense_quant(casc_in, out); }}
''')
                self._decls.append(f"void f{idx}_quant(input_cascade<{cascade_tag}> * __restrict, {stream_type} * __restrict);")

            if split_stage_decls:
                self._decls.extend(split_stage_decls)

            np.testing.assert_array_equal(recon_xt.flatten(), x_tiled)
            np.testing.assert_array_equal(recon_wt.flatten(), k_tiled_full)

        # Connect bytes from the *previous* layer as-is (window size is just count of dtype)
        in_port   = "AIE_IN[0]" if idx == 0 else f"layers[{idx-1}]"
        num_bytes = x_in.size * x_in.itemsize
        with open("model/layer_graph.h", "a") as f:
            if self.input_plios == 1:
                f.write(f"layers[{idx}] = kernel::create(f{idx});\n")
                f.write(f'source(layers[{idx}]) = "layer_{idx}.cc";\n')
                if self.dataflow == 'stream':
                    if self.input_mode == 'stream':
                        f.write(f"auto c{idx} = connect<stream>({in_port}.out[0], layers[{idx}].in[0]);\n")
                    else:
                        assert cascin_kernel is not None
                        f.write(f"kernel {cascin_kernel} = kernel::create({cascin_func});\n")
                        f.write(f'source({cascin_kernel}) = "{cascin_kernel}.cc";\n')
                        f.write(f"runtime<ratio>({cascin_kernel}) = 1.0;\n")
                        prev = f"layers[{idx-1}]"
                        f.write(f"connect<cascade>({prev}.out[0], {cascin_kernel}.in[0]);\n")
                        f.write(f"auto c{idx}_casc = connect<stream>({cascin_kernel}.out[0], layers[{idx}].in[0]);\n")
                elif self.dataflow == 'window':
                    f.write(f"connect<window<{num_bytes}>>({in_port}.out[0], layers[{idx}].in[0]);\n\n")
                if idx == 0 and num_bytes > 32768 and self.input_mode == 'stream':
                    f.write(f"single_buffer(layers[{idx}].in[0]);\n")
            else:
                kernel_names = []
                for tile_idx in range(self.input_plios):
                    handle = f"layer_{idx}_partial{tile_idx}"
                    func = f"f{idx}_partial{tile_idx}"
                    f.write(f"kernel {handle} = kernel::create({func});\n")
                    f.write(f'source({handle}) = "layer_{idx}_partial{tile_idx}.cc";\n')
                    f.write(f"runtime<ratio>({handle}) = 1.0;\n")
                    kernel_names.append(handle)
                if cascin_kernel and self.input_mode == 'stream':
                    pass  # handled above
                if split_stage_names:
                    for stage_idx, stage_kernel in enumerate(split_stage_names):
                        stage_func, has_casc_out = split_stage_funcs[stage_idx]
                        f.write(f"kernel {stage_kernel} = kernel::create({stage_func});\n")
                        f.write(f'source({stage_kernel}) = "{stage_kernel}.cc";\n')
                        f.write(f"runtime<ratio>({stage_kernel}) = 1.0;\n")
                if not self.cascade_out:
                    f.write(f"layers[{idx}] = kernel::create(f{idx}_quant);\n")
                    f.write(f'source(layers[{idx}]) = "layer_{idx}_quant.cc";\n')
                    f.write(f"runtime<ratio>(layers[{idx}]) = 1.0;\n")
                else:
                    f.write(f"layers[{idx}] = {kernel_names[-1]};\n")
                for tile_idx, handle in enumerate(kernel_names):
                    if self.input_mode == 'stream':
                        f.write(f"connect<stream>(AIE_IN[{tile_idx}].out[0], {handle}.in[0]);\n")
                    else:
                        stage_kernel = split_stage_names[tile_idx]
                        stage_func, has_casc_out = split_stage_funcs[tile_idx]
                        stream_port = 1 if has_casc_out else 0
                        f.write(f"connect<stream>({stage_kernel}.out[{stream_port}], {handle}.in[0]);\n")
                    if tile_idx > 0:
                        prev = kernel_names[tile_idx-1]
                        f.write(f"connect<cascade>({prev}.out[0], {handle}.in[1]);\n")
                if split_stage_names:
                    prev_source = f"layers[{idx-1}]"
                    f.write(f"connect<cascade>({prev_source}.out[0], {split_stage_names[0]}.in[0]);\n")
                    for stage_idx, stage_kernel in enumerate(split_stage_names[:-1]):
                        _, has_casc_out = split_stage_funcs[stage_idx]
                        if has_casc_out:
                            next_stage = split_stage_names[stage_idx+1]
                            f.write(f"connect<cascade>({stage_kernel}.out[0], {next_stage}.in[0]);\n")
                if not self.cascade_out:
                    f.write(f"connect<cascade>({kernel_names[-1]}.out[0], layers[{idx}].in[0]);\n")


class Sequential:
    def __init__(self, iterations=1, dtype='i16', dataflow='stream', free=False, input_plios=1, output_plios=1, **kwargs):
        self.layers = []
        self.iterations = iterations
        self.dtype = dtype
        self.dataflow = dataflow
        self.free = free
        self.input_plios = input_plios
        self.output_plios = output_plios

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
        self.output_plios = getattr(last, "output_plios", self.output_plios)
        layout = last.get_output_layout() if hasattr(last, "get_output_layout") else None
        if layout is not None:
            m_tile, n_tile = layout
            if self.output_plios == 1:
                tiled_last = tile_matrix(x, m_tile, n_tile)
                np.savetxt("data/out_ref.txt",
                           np.tile(tiled_last, (self.iterations,1)).reshape(-1,TY_DICT[self.dtype]['per_plio']),
                           fmt="%s", delimiter=" ")
            else:
                cols_per = x.shape[1] // self.output_plios
                for plio in range(self.output_plios):
                    chunk = x[:, plio*cols_per:(plio+1)*cols_per]
                    chunk_tiled = tile_matrix(chunk, m_tile, n_tile)
                    np.savetxt(f"data/out_ref_{plio}.txt",
                               np.tile(chunk_tiled, (self.iterations,1)).reshape(-1,TY_DICT[self.dtype]['per_plio']),
                               fmt="%s", delimiter=" ")
        out_bytes = x.size * x.itemsize

        with open("model/layer_graph.h", "a") as f:
            multi_handles = getattr(last, "_multi_output_handles", None)
            if self.output_plios == 1 or not multi_handles:
                if out_bytes >= 32768:
                    f.write(f"single_buffer(layers[{N_LAYERS-1}].out[0]);\n")

                if self.dataflow == 'stream':
                    f.write(f"auto c{N_LAYERS} = connect<stream>(layers[{N_LAYERS-1}].out[0], AIE_OUT[0].in[0]);\n")
                elif self.dataflow == 'window':
                    f.write(f"connect<window<{out_bytes}>>(layers[{N_LAYERS-1}].out[0], AIE_OUT[0].in[0]);\n")
            else:
                assert len(multi_handles) == self.output_plios, "Mismatch in output PLIO handles"
                for plio, (handle, port_idx) in enumerate(multi_handles):
                    f.write(f"auto c{N_LAYERS}_{plio} = connect<stream>({handle}.out[{port_idx}], AIE_OUT[{plio}].in[0]);\n")

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

class CascadeToStream(Layer):
    def __init__(self, shift=0, relu=False, m_tile=2, n_tile=8, k_tile=8, dtype='i16', free=False, dataflow='stream', **kwargs):
        super().__init__()
        self.shift = shift
        self.relu = relu
        self.m_tile = m_tile
        self.n_tile = n_tile
        self.k_tile = k_tile
        self.dtype = dtype
        self.free = free
        self.dataflow = dataflow
        self.input_mode = 'cascade'
        self.output_mode = 'stream'
        self.output_plios = 1
        self._last_in = None

    def forward(self, x_in):
        self._last_in = x_in
        return x_in

    def get_output_layout(self):
        return (self.m_tile, self.n_tile)

    def emit(self, idx, x_in, y_ref, iterations, layers):
        assert idx > 0, "CascadeToStream requires a previous layer"
        prev_layer = layers[idx-1]
        assert getattr(prev_layer, "output_mode", "stream") == "cascade", "Previous layer must output cascade"

        x2d = self._last_in if self._last_in is not None else x_in
        mm_M = x2d.shape[0]
        mm_N = x2d.shape[1]

        ty_str = TY_DICT[self.dtype]['str']
        cascade_tag = {'i8': 'acc32', 'i16': 'acc48'}[self.dtype]

        with open(f"model/layer_{idx}.cc", "w") as f:
            if self.free:
                f.write(f'#define FREE\n')
            f.write(f"#define DENSE_CASC_TYPE {cascade_tag}\n")
            f.write(f'''\
#define DTYPE {ty_str}
#define mm_m {self.m_tile}
#define mm_k {self.k_tile}
#define mm_n {self.n_tile}
#define mm_M {mm_M}
#define mm_K {self.k_tile}
#define mm_N {mm_N}
#define SHIFT {self.shift}
#define DO_RELU {str(self.relu).lower()}

#include <cstdint>
__attribute__((section(".data"))) alignas(32) {ty_str}_t matB[1] = {{ 0 }};

#include "dense_stream.h"

void f{idx}(input_cascade<{cascade_tag}> * __restrict casc_in, output_{self.dataflow}_{ty_str} * __restrict out){{ dense_quant(casc_in, out); }}
''')
        self._decls = [f"void f{idx}(input_cascade<{cascade_tag}> * __restrict, output_{self.dataflow}_{ty_str} * __restrict);"]

        with open("model/layer_graph.h", "a") as f:
            f.write(f"layers[{idx}] = kernel::create(f{idx});\n")
            f.write(f'source(layers[{idx}]) = "layer_{idx}.cc";\n')
            f.write(f"runtime<ratio>(layers[{idx}]) = 1.0;\n")
            prev = f"layers[{idx-1}]"
            f.write(f"connect<cascade>({prev}.out[0], layers[{idx}].in[0]);\n")
