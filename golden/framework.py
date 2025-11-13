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

def write_plio_file(path, vec, per_plio, iterations):
    base = vec.astype(vec.dtype, copy=False)
    pad = (-base.size) % per_plio
    if pad:
        base = np.concatenate([base, np.zeros(pad, dtype=base.dtype)])
    tiled = np.tile(base, iterations)
    tiled = tiled.reshape(-1, per_plio)
    np.savetxt(path, tiled, fmt="%s", delimiter=" ")
    return pad

class Layer:
    def emit(self, idx, x_in, y_ref, iterations, layers):
        raise NotImplementedError
    def forward(self, x_in):
        raise NotImplementedError

class Dense(Layer):
    """
    Dense operating row-wise on (R, K) -> (R, N); AIE tiles m=2, k=8, n=8.
    Accepts either 2-D (R,K) or 3-D NHWC (H,W,C). For 3-D, it flattens to (H*W, C).
    """
    def __init__(self, N, shift=0, relu=False, m_tile=2, k_tile=8, n_tile=8, dtype='i16', dataflow='stream', free=False, **kwargs):
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

        # weights tiled KxN -> k_tiled
        k_tiled = tile_matrix(self.W, k, n)

        # IO files (dense uses tiled dumps)
        per_plio = TY_DICT[self.dtype]['per_plio']
        x_tiled = tile_matrix(x2d, m, k)
        a_tiled = tile_matrix(y_ref, m, n)
        np.savetxt(f"data/x{idx}.txt", np.tile(x_tiled, (iterations, 1)).reshape(-1, per_plio), fmt="%s", delimiter=" ")
        np.savetxt(f"data/a{idx}.txt", np.tile(a_tiled, (iterations, 1)).reshape(-1, per_plio), fmt="%s", delimiter=" ")

        ty_str = TY_DICT[self.dtype]['str']

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
__attribute__((section(".data"))) alignas(32) {ty_str}_t matB [{k_tiled.size}] = {{ {", ".join(str(int(x)) for x in k_tiled)} }};

#include "dense_{self.dataflow}.h"

void f{idx}(input_{self.dataflow}_{ty_str} * __restrict in, output_{self.dataflow}_{ty_str} * __restrict out){{ dense(in, out);}}
''')

        # Connect bytes from the *previous* layer as-is (window size is just count of dtype)
        in_port   = "AIE_IN" if idx == 0 else f"layers[{idx-1}]"
        num_bytes = x_in.size * x_in.itemsize
        with open("model/layer_graph.h", "a") as f:
            f.write(f"layers[{idx}] = kernel::create(f{idx});\n")
            f.write(f'source(layers[{idx}]) = "layer_{idx}.cc";\n')
            if self.dataflow == 'stream':
                f.write(f"auto c{idx} = connect<stream>({in_port}.out[0], layers[{idx}].in[0]);\n")
                # f.write(f"fifo_depth(c{idx}) = 32;\n")
            elif self.dataflow == 'window':
                f.write(f"connect<window<{num_bytes}>>({in_port}.out[0], layers[{idx}].in[0]);\n\n")
            if idx == 0 and num_bytes > 32768:
                f.write(f"single_buffer(layers[{idx}].in[0]);\n")


class Conv2d(Layer):
    def __init__(self, XH, XW, CI, CO, KH, KW,
                 stride=(1, 1), padding=(0, 0),
                 shift=0, relu=False, dtype='i8', dataflow='stream', free=False, **kwargs):
        assert dataflow == 'stream', "Conv2d stream kernel only supports stream dataflow"
        assert dtype == 'i8', "Conv2d stream kernel currently supports int8"
        self.XH = XH
        self.XW = XW
        self.CI = CI
        self.CO = CO
        self.KH = KH
        self.KW = KW
        self.SH, self.SW = stride
        self.PH, self.PW = padding
        assert self.SH > 0 and self.SW > 0
        numer_h = self.XH + 2 * self.PH - self.KH
        numer_w = self.XW + 2 * self.PW - self.KW
        assert numer_h >= 0 and numer_w >= 0, "Invalid padding/kernel combo"
        assert (numer_h % self.SH) == 0, "Stride/padding mismatch on height"
        assert (numer_w % self.SW) == 0, "Stride/padding mismatch on width"
        self.YH = numer_h // self.SH + 1
        self.YW = numer_w // self.SW + 1
        self.shift = shift
        self.relu = relu
        self.dtype = dtype
        self.dataflow = dataflow
        self.free = free
        self.per_plio = TY_DICT[self.dtype]['per_plio']
        self.W = None
        self._last_input = None
        self.pad_in = 0
        self.default_pad_out = (- (self.YH * self.YW * self.CO)) % self.per_plio
        self.pad_out = self.default_pad_out

    def forward(self, x_in):
        assert x_in.ndim == 3, "Conv2d expects NHWC input"
        assert x_in.shape == (self.XH, self.XW, self.CI), f"Expected {(self.XH, self.XW, self.CI)}, got {x_in.shape}"
        self._last_input = x_in
        self.W = np.random.randint(0, 128, size=(self.KH, self.KW, self.CI, self.CO), dtype=np.int8)

        y = np.zeros((self.YH, self.YW, self.CO), dtype=np.int32)
        for oh in range(self.YH):
            for ow in range(self.YW):
                acc = np.zeros((self.CO,), dtype=np.int32)
                for kh in range(self.KH):
                    ih = oh * self.SH - self.PH + kh
                    if ih < 0 or ih >= self.XH:
                        continue
                    for kw in range(self.KW):
                        iw = ow * self.SW - self.PW + kw
                        if iw < 0 or iw >= self.XW:
                            continue
                        act = x_in[ih, iw, :].astype(np.int32)
                        w = self.W[kh, kw].astype(np.int32)
                        acc += act @ w
                y[oh, ow] = acc

        if self.shift > 0:
            y = (y + (1 << (self.shift - 1))) >> self.shift
        y = y.astype(TY_DICT[self.dtype]['np'])
        if self.relu:
            y = np.maximum(0, y)
        return y

    def _flatten_stream(self, array):
        return array.astype(TY_DICT[self.dtype]['np']).flatten()

    def _pack_weights(self):
        ci_aligned = ((self.CI + 7) // 8) * 8
        xc8 = ci_aligned // 8
        yc8 = ((self.CO + 7) // 8)
        co_aligned = yc8 * 8
        packed = np.zeros((self.KH, self.KW, ci_aligned, co_aligned), dtype=np.int8)
        packed[:, :, :self.CI, :self.CO] = self.W
        tiles = []
        for xc in range(xc8):
            for yc in range(yc8):
                for kh in range(self.KH):
                    for kw in range(self.KW):
                        block = packed[kh, kw, xc*8:(xc+1)*8, yc*8:(yc+1)*8]
                        tiles.append(block.flatten())
        return np.concatenate(tiles)

    def emit(self, idx, x_in, y_ref, iterations, layers):
        per_plio = TY_DICT[self.dtype]['per_plio']
        ty_str = TY_DICT[self.dtype]['str']

        x_flat = self._flatten_stream(self._last_input if self._last_input is not None else x_in)
        y_flat = self._flatten_stream(y_ref)

        pad_written = write_plio_file(f"data/x{idx}.txt", x_flat, per_plio, iterations)
        write_plio_file(f"data/a{idx}.txt", y_flat, per_plio, iterations)
        if idx == 0:
            self.pad_in = pad_written

        weights = self._pack_weights()
        weight_str = ", ".join(str(int(v)) for v in weights)

        pad_in_macro = self.pad_in
        pad_out_macro = self.pad_out

        with open(f"model/layer_{idx}.cc", "a") as f:
            if self.free:
                f.write(f'#define FREE\n')
            f.write(f'''
#define DTYPE {ty_str}
#define KH {self.KH}
#define KW {self.KW}
#define CI {self.CI}
#define CO {self.CO}
#define XH {self.XH}
#define XW {self.XW}
#define YH {self.YH}
#define YW {self.YW}
#define SH {self.SH}
#define SW {self.SW}
#define PH {self.PH}
#define PW {self.PW}
#define SHIFT {self.shift}
#define DO_RELU {str(self.relu).lower()}
#define PAD_IN {pad_in_macro}
#define PAD_OUT {pad_out_macro}

#include <cstdint>
__attribute__((section(".data"))) alignas(32) const int8_t k_p [{weights.size}] = {{ {weight_str} }};

#include "conv_stream.h"

void f{idx}(input_stream_int8 * __restrict in, output_stream_int8 * __restrict out){{ conv_stream(in, out);}}
''')

        in_port   = "AIE_IN" if idx == 0 else f"layers[{idx-1}]"
        num_bytes = x_in.size * x_in.itemsize
        with open("model/layer_graph.h", "a") as f:
            f.write(f"layers[{idx}] = kernel::create(f{idx});\n")
            f.write(f'source(layers[{idx}]) = "layer_{idx}.cc";\n')
            f.write(f"auto c{idx} = connect<stream>({in_port}.out[0], layers[{idx}].in[0]);\n")
            if idx == 0 and num_bytes > 32768:
                f.write(f"single_buffer(layers[{idx}].in[0]);\n")


class Sequential:
    def __init__(self, iterations=1, dtype='i16', dataflow='stream', free=False, **kwargs):
        self.layers = []
        self.iterations = iterations
        self.dtype = dtype
        self.dataflow = dataflow
        self.free = free

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

        per_plio = TY_DICT[self.dtype]['per_plio']
        pad_carry = (-(x0.size) % per_plio)
        x = x0
        for idx, layer in enumerate(self.layers):
            if isinstance(layer, Conv2d):
                layer.pad_in = pad_carry
                if layer.pad_out is None:
                    layer.pad_out = layer.default_pad_out
            y = layer.forward(x)
            layer.emit(idx, x, y, self.iterations, self.layers)
            x = y
            if isinstance(layer, Conv2d):
                pad_carry = layer.pad_out

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
        elif isinstance(last, Conv2d):
            flat_last = x.flatten()
            write_plio_file("data/out_ref.txt", flat_last, TY_DICT[self.dtype]['per_plio'], self.iterations)
            out_bytes = x.size * x.itemsize
        else:
            flat_last = x.flatten()
            np.savetxt("data/out_ref.txt",
                       np.tile(flat_last, (self.iterations,1)).reshape(-1,TY_DICT[self.dtype]['per_plio']),
                       fmt="%s", delimiter=" ")
            out_bytes = x.size * x.itemsize

        with open("model/layer_graph.h", "a") as f:
            if out_bytes >= 32768:
                f.write(f"single_buffer(layers[{N_LAYERS-1}].out[0]);\n")

            if self.dataflow == 'stream':
                f.write(f"auto c{N_LAYERS} = connect<stream>(layers[{N_LAYERS-1}].out[0], AIE_OUT.in[0]);\n")
                # f.write(f"fifo_depth(c{N_LAYERS}) = 32;\n")
            elif self.dataflow == 'window':
                f.write(f"connect<window<{out_bytes}>>(layers[{N_LAYERS-1}].out[0], AIE_OUT.in[0]);\n")

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
            for idx, layer in enumerate(self.layers):
                f.write(f'void f{idx}(input_{self.dataflow}_{ty_str} * __restrict, output_{self.dataflow}_{ty_str} * __restrict);\n')
        return x
