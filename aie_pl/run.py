#!/usr/bin/env python3

import numpy as np
import os, subprocess, json, sysconfig

# ── Knobs ──────────────────────────────────────────────────────────────────────
BATCH        = 8
SHIFT        = 0
N_ITER       = 1

# AIE knobs (shared across all AIE layers)
CAS_LENGTH   = 1
CAS_NUM      = 1
TILE_M, TILE_K, TILE_N = 4, 8, 8

# HLS knobs (for synthesis reference)
REUSE_FACTOR = 1
# ──────────────────────────────────────────────────────────────────────────────


class Dense:
    def __init__(self, n_in, n_out, target='pl'):
        assert target in ('pl', 'aie'), f"target must be 'pl' or 'aie', got {target!r}"
        self.n_in   = n_in
        self.n_out  = n_out
        self.target = target
        self.W = None
        self.b = None

    def init_weights(self, rng):
        self.W = rng.integers(-3, 4, (self.n_in, self.n_out), dtype=np.int8)
        self.b = np.zeros(self.n_out, dtype=np.int32)

    def forward(self, x, shift):
        acc = x.astype(np.int32) @ self.W.astype(np.int32) + self.b.astype(np.int32)
        out = np.maximum(0, acc >> shift)
        if self.target == 'aie':
            return np.clip(out, 0, 255).astype(np.uint8)
        else:
            return np.clip(out, 0, 127).astype(np.int8)


class Model:
    def __init__(self, layers, batch=BATCH, shift=SHIFT, n_iter=N_ITER,
                 cas_length=CAS_LENGTH, cas_num=CAS_NUM,
                 tile_m=TILE_M, tile_k=TILE_K, tile_n=TILE_N,
                 reuse_factor=REUSE_FACTOR):
        self.layers       = layers
        self.batch        = batch
        self.shift        = shift
        self.n_iter       = n_iter
        self.cas_length   = cas_length
        self.cas_num      = cas_num
        self.tile_m       = tile_m
        self.tile_k       = tile_k
        self.tile_n       = tile_n
        self.reuse_factor = reuse_factor

        # Validate consecutive layer sizes
        for i in range(len(layers) - 1):
            assert layers[i].n_out == layers[i+1].n_in, \
                f"Layer {i} output {layers[i].n_out} != Layer {i+1} input {layers[i+1].n_in}"

        # Validate AIE tiling constraints (across all AIE layers)
        has_aie = any(l.target == 'aie' for l in layers)
        if has_aie:
            assert batch % (2 * tile_m) == 0, \
                f"BATCH={batch} not divisible by 2*M={2*tile_m}"
            for i, l in enumerate(layers):
                if l.target != 'aie':
                    continue
                in_sl  = l.n_in  // cas_length
                out_sl = l.n_out // cas_num
                assert in_sl  % (2 * tile_k) == 0, \
                    f"AIE layer {i}: IN/CAS_LEN={in_sl} not divisible by 2*K={2*tile_k}"
                assert out_sl % (2 * tile_n) == 0, \
                    f"AIE layer {i}: OUT/CAS_NUM={out_sl} not divisible by 2*N={2*tile_n}"
                assert (batch * l.n_in)  % 16 == 0, \
                    f"AIE layer {i}: BATCH*IN={batch*l.n_in} not divisible by 16 (PLIO)"
                assert (batch * l.n_out) % 16 == 0, \
                    f"AIE layer {i}: BATCH*OUT={batch*l.n_out} not divisible by 16 (PLIO)"

    def init_weights(self, seed=42):
        rng = np.random.default_rng(seed)
        for layer in self.layers:
            layer.init_weights(rng)

    def numpy_forward(self, x):
        for layer in self.layers:
            x = layer.forward(x, self.shift)
        return x

    # ── Code / file generation ─────────────────────────────────────────────────

    def _cfg(self, i):  return f"A{i}Cfg"
    def _wts(self, i):  return f"weights_a{i}"
    def _bias(self, i): return f"bias_a{i}"

    def _pack_weights(self, W):
        K, N = self.tile_k, self.tile_n
        KF, NF = W.shape
        return W.reshape(KF // K, K, NF // N, N).transpose(0, 2, 1, 3).flatten()

    def _write_weight_header(self, path, name, W_packed, in_sl, out_sl):
        cn, cl = self.cas_num, self.cas_length
        W3 = W_packed.reshape(cn, cl, in_sl * out_sl)
        with open(path, "w") as f:
            f.write(f"int8_t {name}[{cn}][{cl}][{in_sl * out_sl}] = {{\n")
            for ch in range(cn):
                f.write("  {\n")
                for col in range(cl):
                    vals = ", ".join(str(int(v)) for v in W3[ch][col])
                    f.write(f"    {{{vals}}}\n")
                f.write("  },\n")
            f.write("};\n")

    def _write_bias_header(self, path, name, b, out_sl):
        b2 = b.reshape(self.cas_num, out_sl)
        with open(path, "w") as f:
            f.write(f"int32_t {name}[{self.cas_num}][{out_sl}] = {{\n")
            for ch in range(self.cas_num):
                vals = ", ".join(str(int(v)) for v in b2[ch])
                f.write(f"  {{{vals}}},\n")
            f.write("};\n")

    def _aie_groups(self):
        """Returns list of groups; each group is a list of model-layer indices."""
        groups, i = [], 0
        while i < len(self.layers):
            if self.layers[i].target == 'aie':
                start = i
                while i < len(self.layers) and self.layers[i].target == 'aie':
                    i += 1
                groups.append(list(range(start, i)))
            else:
                i += 1
        return groups

    def _parameters_h(self, group_layers):
        lines = [
            "#pragma once",
            "#include <adf.h>",
            "#include <aie_api/aie.hpp>",
            "#include <cstdint>",
            "",
            f"#define N_ITER {self.n_iter}",
            "",
        ]
        for i, l in enumerate(group_layers):
            in_sl  = l.n_in  // self.cas_length
            out_sl = l.n_out // self.cas_num
            col    = 1 + 2 * i
            data_t = "int8_t" if i == 0 else "uint8_t"
            name   = self._cfg(i)
            lines += [
                f"struct {name} {{",
                f"  using data_t       = {data_t};",
                f"  using weight_t     = int8_t;",
                f"  using result_t     = uint8_t;",
                f"  using bias_t       = int32_t;",
                f"  using acc_scalar_t = acc32;",
                f"  static constexpr int IN_FEAT  = {l.n_in};",
                f"  static constexpr int OUT_FEAT = {l.n_out};",
                f"  static constexpr int CAS_LENGTH = {self.cas_length};",
                f"  static constexpr int CAS_NUM    = {self.cas_num};",
                f"  static constexpr bool USE_BIAS        = true;",
                f"  static constexpr bool USE_RELU        = true;",
                f"  static constexpr bool TRANSPOSE_INPUT = false;",
                f"  static constexpr int SHIFT = {self.shift};",
                f"  static constexpr int M = {self.tile_m}, K = {self.tile_k}, N = {self.tile_n};",
                f"  static constexpr int col_placement = {col}, row_placement = 0;",
                f"  static constexpr int padded_independent_extent = {self.batch};",
                f"  static constexpr int padded_IN_FEAT  = {l.n_in};",
                f"  static constexpr int padded_OUT_FEAT = {l.n_out};",
                f"  static constexpr int IN_FEAT_SLICE   = {in_sl};",
                f"  static constexpr int OUT_FEAT_SLICE  = {out_sl};",
                f"  static constexpr int RAW_IN_FEAT_SLICE  = {in_sl};",
                f"  static constexpr int RAW_OUT_FEAT_SLICE = {out_sl};",
                f"#if __cplusplus >= 202002L",
                f"  static constexpr auto ROUNDING   = aie::rounding_mode::conv_even;",
                f"  static constexpr auto SATURATION = aie::saturation_mode::saturate;",
                f"#endif",
                f'  static constexpr const char* ROUNDING_TOKEN   = "conv_even";',
                f'  static constexpr const char* SATURATION_TOKEN = "saturate";',
                f"}};",
                "",
            ]
        return "\n".join(lines)

    def _graph_cpp(self, group_layers):
        n    = len(group_layers)
        cfg  = self._cfg
        wts  = self._wts
        bias = self._bias
        L    = []

        def add(*args): L.extend(args)

        def buf_write_tiled(c, port):
            add(
                f"    write_access({port}) = tiling({{",
                f"      .buffer_dimension = {{ {c}::OUT_FEAT, {c}::padded_independent_extent }},",
                f"      .tiling_dimension = {{ {c}::N, {c}::M }},",
                f"      .offset = {{ 0, 0 }},",
                f"      .tile_traversal = {{",
                f"        {{ .dimension = 0, .stride = {c}::N, .wrap = {c}::OUT_FEAT / {c}::N }},",
                f"        {{ .dimension = 1, .stride = {c}::M, .wrap = {c}::padded_independent_extent / {c}::M }}",
                f"      }}",
                f"    }});",
            )

        def buf_read_tiled(c, port):
            add(
                f"    read_access({port}) = tiling({{",
                f"      .buffer_dimension = {{ {c}::IN_FEAT, {c}::padded_independent_extent }},",
                f"      .tiling_dimension = {{ {c}::K, {c}::M }},",
                f"      .offset = {{ 0, 0 }},",
                f"      .tile_traversal = {{",
                f"        {{ .dimension = 0, .stride = {c}::K, .wrap = {c}::IN_FEAT / {c}::K }},",
                f"        {{ .dimension = 1, .stride = {c}::M, .wrap = {c}::padded_independent_extent / {c}::M }}",
                f"      }},",
                f"      .boundary_dimension = {{ {c}::IN_FEAT, {c}::padded_independent_extent }}",
                f"    }});",
            )

        # ── Includes ──
        add(
            '#include <adf.h>',
            '#include <fstream>',
            '#include "parameters.h"',
            '#include "dense_graph.h"',
            '',
            'extern "C" {',
        )
        for i in range(n):
            add(f'  #include "weights/{wts(i)}.h"',
                f'  #include "weights/{bias(i)}.h"')
        add('}', '', 'using namespace adf;', '')

        # ── top_graph ──
        add('class top_graph : public graph {', 'public:',
            '  input_port  ifm[1];', '  output_port ofm[1];', '')
        for i in range(n):
            add(f'  input_port wts{i} [{cfg(i)}::CAS_NUM * {cfg(i)}::CAS_LENGTH];',
                f'  input_port bias{i}[{cfg(i)}::CAS_NUM];')
        add('', 'private:')
        for i in range(n):
            add(f'  dense_bias_relu_graph<{cfg(i)}> l{i};')
        add('')
        add(f'  shared_buffer<typename {cfg(0)}::data_t>    buffer_in;')
        for i in range(n - 1):
            add(f'  shared_buffer<typename {cfg(i)}::result_t>  buffer_mid{i};')
        add(f'  shared_buffer<typename {cfg(n-1)}::result_t> buffer_out;')
        add('', 'public:', '  top_graph() {')

        # buffer_in
        c0 = cfg(0)
        add(
            f'    buffer_in = shared_buffer<typename {c0}::data_t>::create(',
            f'      {{ {c0}::IN_FEAT, {c0}::padded_independent_extent }}, 1, 1);',
            f'    num_buffers(buffer_in) = 2;',
            f'    connect<>(ifm[0], buffer_in.in[0]);',
            f'    write_access(buffer_in.in[0]) = tiling({{',
            f'      .buffer_dimension = {{ {c0}::IN_FEAT, {c0}::padded_independent_extent }},',
            f'      .tiling_dimension = {{ {c0}::IN_FEAT, {c0}::padded_independent_extent }},',
            f'      .offset = {{ 0, 0 }}',
            f'    }});',
        )
        buf_read_tiled(c0, 'buffer_in.out[0]')
        add(f'    connect<>(buffer_in.out[0], l0.in1[0]);', '')

        # intermediate buffers
        for i in range(n - 1):
            ci, ci1 = cfg(i), cfg(i + 1)
            add(
                f'    buffer_mid{i} = shared_buffer<typename {ci}::result_t>::create(',
                f'      {{ {ci}::OUT_FEAT, {ci}::padded_independent_extent }}, 1, 1);',
                f'    num_buffers(buffer_mid{i}) = 2;',
                f'    connect<>(l{i}.out1[0], buffer_mid{i}.in[0]);',
            )
            buf_write_tiled(ci,  f'buffer_mid{i}.in[0]')
            buf_read_tiled(ci1, f'buffer_mid{i}.out[0]')
            add(f'    connect<>(buffer_mid{i}.out[0], l{i+1}.in1[0]);', '')

        # buffer_out
        cl = cfg(n - 1)
        add(
            f'    buffer_out = shared_buffer<typename {cl}::result_t>::create(',
            f'      {{ {cl}::OUT_FEAT, {cl}::padded_independent_extent }}, 1, 1);',
            f'    num_buffers(buffer_out) = 2;',
            f'    connect<>(l{n-1}.out1[0], buffer_out.in[0]);',
        )
        buf_write_tiled(cl, 'buffer_out.in[0]')
        add(
            f'    read_access(buffer_out.out[0]) = tiling({{',
            f'      .buffer_dimension = {{ {cl}::OUT_FEAT, {cl}::padded_independent_extent }},',
            f'      .tiling_dimension = {{ {cl}::OUT_FEAT, {cl}::padded_independent_extent }},',
            f'      .offset = {{ 0, 0 }},',
            f'      .boundary_dimension = {{ {cl}::OUT_FEAT, {cl}::padded_independent_extent }}',
            f'    }});',
            f'    connect<>(buffer_out.out[0], ofm[0]);',
            '',
        )

        # weight / bias connections
        for i in range(n):
            c = cfg(i)
            add(
                f'    for (int ch = 0; ch < {c}::CAS_NUM; ++ch) {{',
                f'      for (int col = 0; col < {c}::CAS_LENGTH; ++col)',
                f'        connect<>(wts{i}[ch * {c}::CAS_LENGTH + col], l{i}.wts[ch * {c}::CAS_LENGTH + col]);',
                f'      connect<>(bias{i}[ch], l{i}.bias[ch]);',
                f'    }}',
            )
        add('')
        for i in range(n):
            add(f'    l{i}.place_graph({cfg(i)}::col_placement, {cfg(i)}::row_placement);')
        add('  }', '};', '')

        # ── dut_graph ──
        add(
            '// ── DUT: PLIO wrappers ──', '',
            'class dut_graph : public graph {', 'public:',
            '  input_plio  plio_in;', '  output_plio plio_out;', '',
        )
        for i in range(n):
            c = cfg(i)
            add(f'  input_port wts{i} [{c}::CAS_NUM * {c}::CAS_LENGTH];',
                f'  input_port bias{i}[{c}::CAS_NUM];')
        add(
            '', '  top_graph dut;', '',
            '  dut_graph() {',
            '    plio_in  = input_plio::create("PLIO_in",  plio_128_bits, "data/ifm.txt");',
            '    plio_out = output_plio::create("PLIO_out", plio_128_bits, "data/out.txt");',
            '',
            '    connect<>(plio_in.out[0], dut.ifm[0]);',
            '    connect<>(dut.ofm[0], plio_out.in[0]);',
            '',
        )
        for i in range(n):
            c = cfg(i)
            add(
                f'    for (int ch = 0; ch < {c}::CAS_NUM; ++ch) {{',
                f'      for (int col = 0; col < {c}::CAS_LENGTH; ++col)',
                f'        connect<>(wts{i}[ch * {c}::CAS_LENGTH + col], dut.wts{i}[ch * {c}::CAS_LENGTH + col]);',
                f'      connect<>(bias{i}[ch], dut.bias{i}[ch]);',
                f'    }}',
            )
        add('  }', '};', '', 'dut_graph dut;', '')

        # ── main ──
        add(
            '#if defined(__AIESIM__) || defined(__X86SIM__)',
            'int main() {',
            '  dut.init();', '',
        )
        for i in range(n):
            c = cfg(i)
            add(
                f'  for (int ch = 0; ch < {c}::CAS_NUM; ++ch) {{',
                f'    for (int col = 0; col < {c}::CAS_LENGTH; ++col) {{',
                f'      int idx = ch * {c}::CAS_LENGTH + col;',
                f'      dut.update(dut.wts{i}[idx], {wts(i)}[ch][col], {c}::IN_FEAT_SLICE * {c}::OUT_FEAT_SLICE);',
                f'    }}',
                f'    dut.update(dut.bias{i}[ch], {bias(i)}[ch], {c}::OUT_FEAT_SLICE);',
                f'  }}',
            )
        add(
            '',
            '#ifdef __AIESIM__',
            '  event::handle h = event::start_profiling(',
            '    dut.plio_in, dut.plio_out, event::io_stream_start_difference_cycles);',
            '#endif',
            '',
            '  dut.run(N_ITER);',
            '  dut.wait();',
            '',
            '#ifdef __AIESIM__',
            '  long long cycles = event::read_profiling(h);',
            '  event::stop_profiling(h);',
            '  std::system("mkdir -p aiesimulator_output/data");',
            '  std::ofstream lf("aiesimulator_output/data/latency.json");',
            '  lf << "{\\"cycles\\": " << cycles << "}\\n";',
            '#endif',
            '',
            '  dut.end();',
            '  return 0;',
            '}',
            '#endif',
        )
        return "\n".join(L) + "\n"

    def generate_files(self, group_layers):
        os.makedirs("data",        exist_ok=True)
        os.makedirs("aie/weights", exist_ok=True)

        with open("aie/parameters.h", "w") as f:
            f.write(self._parameters_h(group_layers))

        for i, l in enumerate(group_layers):
            in_sl  = l.n_in  // self.cas_length
            out_sl = l.n_out // self.cas_num
            self._write_weight_header(
                f"aie/weights/{self._wts(i)}.h", self._wts(i),
                self._pack_weights(l.W), in_sl, out_sl)
            self._write_bias_header(
                f"aie/weights/{self._bias(i)}.h", self._bias(i), l.b, out_sl)

        with open("aie/graph.cpp", "w") as f:
            f.write(self._graph_cpp(group_layers))

    def _write_plio(self, path, x):
        flat = x.flatten().astype(np.int32)
        np.savetxt(path, flat.reshape(-1, 16), fmt="%d")

    def _read_plio(self, path, last_layer):
        with open(path) as f:
            lines = [l for l in f if not l.startswith("T")]
        flat = np.array([int(v) for l in lines for v in l.split()], dtype=np.int32)
        return flat.reshape(self.batch, last_layer.n_out).astype(np.uint8)

    def run(self, x0):
        groups = self._aie_groups()

        _env = os.environ.copy()
        _conda_lib = sysconfig.get_path("stdlib").rsplit("/lib/", 1)[0] + "/lib"
        _env["LD_LIBRARY_PATH"] = _conda_lib + (":" + _env["LD_LIBRARY_PATH"] if _env.get("LD_LIBRARY_PATH") else "")

        x         = x0.copy()
        gi        = 0       # AIE group index
        total_aie_cycles = 0
        total_aie_layers = 0

        i = 0
        while i < len(self.layers):
            l = self.layers[i]
            if l.target == 'pl':
                print(f"[PL] Layer {i}: dense {l.n_in}→{l.n_out} relu")
                x = l.forward(x, self.shift)
                i += 1
            else:
                group = groups[gi]; gi += 1
                group_layers = [self.layers[mi] for mi in group]

                self.generate_files(group_layers)
                self._write_plio("data/ifm.txt", x)

                print(f"\n[AIE] Group {gi} (layers {group[0]}–{group[-1]}): {len(group)} layer(s) (make sim)...")
                subprocess.run(["make", "sim"], check=True, env=_env)

                aie_out = self._read_plio("aiesimulator_output/data/out.txt", group_layers[-1])

                # Verify against numpy reference for this group
                x_ref = x.copy()
                for mi in group:
                    x_ref = self.layers[mi].forward(x_ref, self.shift)

                if np.array_equal(aie_out, x_ref):
                    print(f"  ✓ matches reference {aie_out.shape}")
                else:
                    diff = np.abs(aie_out.astype(np.int32) - x_ref.astype(np.int32))
                    print(f"  ✗ mismatch: max diff={diff.max()}, mean={diff.mean():.3f}")
                    print(f"    sim[0]: {aie_out[0]}")
                    print(f"    ref[0]: {x_ref[0]}")

                latency_path = "aiesimulator_output/data/latency.json"
                if os.path.exists(latency_path):
                    with open(latency_path) as f:
                        total_aie_cycles += json.load(f)["cycles"]
                total_aie_layers += len(group)

                x = aie_out
                i = group[-1] + 1

        # Latency report
        print("\n=== Latency Report ===")
        AIE_FREQ_GHZ = 1.25
        PL_FREQ_MHZ  = 312.5
        pl_layers = [l for l in self.layers if l.target == 'pl']
        pl_cycles = sum(self.batch * l.n_in // self.reuse_factor for l in pl_layers)

        if total_aie_cycles:
            aie_ns = total_aie_cycles / AIE_FREQ_GHZ
            print(f"AIE latency ({total_aie_layers} layers, {len(groups)} group(s)): "
                  f"{total_aie_cycles} cycles  ({aie_ns:.1f} ns @ {AIE_FREQ_GHZ} GHz)")

        pl_ns = pl_cycles / (PL_FREQ_MHZ * 1e6) * 1e9
        print(f"PL  latency ({len(pl_layers)} layers): ~{pl_cycles} cycles  ({pl_ns:.1f} ns @ {PL_FREQ_MHZ} MHz, estimated)")

        if total_aie_cycles:
            total_ns = aie_ns + pl_ns
            tp_m     = self.batch / (total_ns * 1e-9) / 1e6
            print(f"Total latency:  {total_ns:.1f} ns")
            print(f"Throughput:     {tp_m:.2f} M samples/sec")

        return x


# ── Model definition ──────────────────────────────────────────────────────────

model = Model([
    Dense(64, 64, target='aie'),
    Dense(64, 64, target='pl'),
    Dense(64, 64, target='aie'),
    Dense(64, 64, target='pl'),
])
model.init_weights()

x0 = np.random.default_rng(0).integers(0, 4, (BATCH, model.layers[0].n_in), dtype=np.int8)
model.run(x0)
