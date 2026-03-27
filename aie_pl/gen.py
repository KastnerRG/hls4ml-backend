#!/usr/bin/env python3
"""
gen.py — generate all build artefacts for the AIE+PL dense NN.

Invoked by Make when knobs or the model change. Does not call any tools.
Edit LAYERS to change the network architecture.
"""
import argparse, io, json, os
import numpy as np

# ── Network definition ────────────────────────────────────────────────────────
# Each entry: ("aie" | "pl", n_in, n_out)
LAYERS = [
    ("pl",  128, 128),
    ("pl",  128, 128),
    ("pl",  128, 128),
    ("pl",  128, 128),
    ("aie", 128, 128),
    ("aie", 128, 128),
    ("aie", 128, 128),
    ("aie", 128, 128),
    ("aie", 128, 128),
    ("aie", 128, 128),
    ("aie", 128, 128),
    ("aie", 128, 128),
    ("pl",  128, 128),
    ("pl",  128, 128),
    ("pl",  128, 128),
    ("pl",  128, 128),
]

# ── Utilities ─────────────────────────────────────────────────────────────────

def write_if_changed(path, content):
    try:
        if open(path).read() == content:
            return
    except FileNotFoundError:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        f.write(content)

def segments():
    segs, gi, pi, i = [], 0, 0, 0
    while i < len(LAYERS):
        t = LAYERS[i][0]; start = i
        while i < len(LAYERS) and LAYERS[i][0] == t:
            i += 1
        if t == 'aie':
            segs.append(('aie', gi, list(range(start, i)))); gi += 1
        else:
            segs.append(('pl',  pi, list(range(start, i)))); pi += 1
    return segs

def pack_weights(W, K, N):
    KF, NF = W.shape
    return W.reshape(KF//K, K, NF//N, N).transpose(0, 2, 1, 3).flatten()

# ── Forward pass ──────────────────────────────────────────────────────────────

def forward(x0, weights, biases, segs, shift):
    aie_inputs, aie_refs = {}, {}
    x = x0.copy()
    for t, gi, idxs in segs:
        if t == 'aie':
            aie_inputs[gi] = x.copy()
            for i in idxs:
                acc = x.astype(np.int32) @ weights[i].astype(np.int32) + biases[i]
                x = np.clip(np.maximum(0, acc >> shift), 0, 255).astype(np.uint8)
            aie_refs[gi] = x.copy()
        else:
            for i in idxs:
                acc = x.astype(np.int32) @ weights[i].astype(np.int32) + biases[i]
                x = np.clip(np.maximum(0, acc >> shift), 0, 127).astype(np.int8)
    return aie_inputs, aie_refs, x

# ── AIE code generation ───────────────────────────────────────────────────────

def parameters_h(aie_groups, a):
    L = ["#pragma once", "#include <adf.h>", "#include <aie_api/aie.hpp>",
         "#include <cstdint>", "", f"#define N_ITER {a.n_iter}", ""]
    col_offset = 0
    for gi, group in enumerate(aie_groups):
        for li, (_, n_in, n_out) in enumerate(group):
            in_sl  = n_in  // a.cas_length
            out_sl = n_out // a.cas_num
            col    = 1 + col_offset + 2 * li
            L += [
                f"struct A{gi}L{li}Cfg {{",
                f"  using data_t       = {'int8_t' if li == 0 else 'uint8_t'};",
                f"  using weight_t     = int8_t;",
                f"  using result_t     = uint8_t;",
                f"  using bias_t       = int32_t;",
                f"  using acc_scalar_t = acc32;",
                f"  static constexpr int IN_FEAT  = {n_in};",
                f"  static constexpr int OUT_FEAT = {n_out};",
                f"  static constexpr int CAS_LENGTH = {a.cas_length};",
                f"  static constexpr int CAS_NUM    = {a.cas_num};",
                f"  static constexpr bool USE_BIAS        = true;",
                f"  static constexpr bool USE_RELU        = true;",
                f"  static constexpr bool TRANSPOSE_INPUT = false;",
                f"  static constexpr int SHIFT = {a.shift};",
                f"  static constexpr int M = {a.tile_m}, K = {a.tile_k}, N = {a.tile_n};",
                f"  static constexpr int col_placement = {col}, row_placement = 0;",
                f"  static constexpr int padded_independent_extent = {a.batch};",
                f"  static constexpr int padded_IN_FEAT  = {n_in};",
                f"  static constexpr int padded_OUT_FEAT = {n_out};",
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
                f"}};", "",
            ]
        col_offset += len(group) * 2
    return "\n".join(L)

def aie_weight_header(name, W_packed, cas_num, cas_length, in_sl, out_sl):
    W3 = W_packed.reshape(cas_num, cas_length, in_sl * out_sl)
    L = [f"int8_t {name}[{cas_num}][{cas_length}][{in_sl * out_sl}] = {{"]
    for ch in range(cas_num):
        L.append("  {")
        for col in range(cas_length):
            L.append(f"    {{{', '.join(str(int(v)) for v in W3[ch][col])}}}")
        L.append("  },")
    return "\n".join(L) + "\n};\n"

def aie_bias_header(name, b, cas_num, out_sl):
    b2 = b.reshape(cas_num, out_sl)
    L = [f"int32_t {name}[{cas_num}][{out_sl}] = {{"]
    for ch in range(cas_num):
        L.append(f"  {{{', '.join(str(int(v)) for v in b2[ch])}}},")
    return "\n".join(L) + "\n};\n"

def graph_cpp(aie_groups, a):
    L = []
    def add(*args): L.extend(args)

    def buf_write_tiled(c, port):
        add(f"    write_access({port}) = tiling({{",
            f"      .buffer_dimension = {{ {c}::OUT_FEAT, {c}::padded_independent_extent }},",
            f"      .tiling_dimension = {{ {c}::N, {c}::M }},",
            f"      .offset = {{ 0, 0 }},",
            f"      .tile_traversal = {{",
            f"        {{ .dimension = 0, .stride = {c}::N, .wrap = {c}::OUT_FEAT / {c}::N }},",
            f"        {{ .dimension = 1, .stride = {c}::M, .wrap = {c}::padded_independent_extent / {c}::M }}",
            f"      }}", f"    }});")

    def buf_read_tiled(c, port):
        add(f"    read_access({port}) = tiling({{",
            f"      .buffer_dimension = {{ {c}::IN_FEAT, {c}::padded_independent_extent }},",
            f"      .tiling_dimension = {{ {c}::K, {c}::M }},",
            f"      .offset = {{ 0, 0 }},",
            f"      .tile_traversal = {{",
            f"        {{ .dimension = 0, .stride = {c}::K, .wrap = {c}::IN_FEAT / {c}::K }},",
            f"        {{ .dimension = 1, .stride = {c}::M, .wrap = {c}::padded_independent_extent / {c}::M }}",
            f"      }},",
            f"      .boundary_dimension = {{ {c}::IN_FEAT, {c}::padded_independent_extent }}",
            f"    }});")

    add('#include <adf.h>', '#include <fstream>', '#include "parameters.h"',
        '#include "dense_graph.h"', '', 'extern "C" {')
    for gi, group in enumerate(aie_groups):
        for li in range(len(group)):
            add(f'  #include "weights/g{gi}_w{li}.h"',
                f'  #include "weights/g{gi}_b{li}.h"')
    add('}', '', 'using namespace adf;', '')

    add('class top_graph : public graph {', 'public:')
    for gi in range(len(aie_groups)):
        add(f'  input_port  g{gi}_ifm[1];', f'  output_port g{gi}_ofm[1];')
    add('')
    for gi, group in enumerate(aie_groups):
        for li in range(len(group)):
            c = f"A{gi}L{li}Cfg"
            add(f'  input_port g{gi}_wts{li}[{c}::CAS_NUM * {c}::CAS_LENGTH];',
                f'  input_port g{gi}_bias{li}[{c}::CAS_NUM];')
    add('', 'private:')
    for gi, group in enumerate(aie_groups):
        for li in range(len(group)):
            add(f'  dense_bias_relu_graph<A{gi}L{li}Cfg> g{gi}_l{li};')
    add('')
    for gi, group in enumerate(aie_groups):
        c0 = f"A{gi}L0Cfg"; cl = f"A{gi}L{len(group)-1}Cfg"
        add(f'  shared_buffer<typename {c0}::data_t>    g{gi}_buf_in;')
        for li in range(len(group) - 1):
            add(f'  shared_buffer<typename A{gi}L{li}Cfg::result_t>  g{gi}_buf_mid{li};')
        add(f'  shared_buffer<typename {cl}::result_t> g{gi}_buf_out;')

    add('', 'public:', '  top_graph() {')
    for gi, group in enumerate(aie_groups):
        n = len(group); c0 = f"A{gi}L0Cfg"; cl = f"A{gi}L{n-1}Cfg"
        add(f'    // ── Group {gi} ──',
            f'    g{gi}_buf_in = shared_buffer<typename {c0}::data_t>::create(',
            f'      {{ {c0}::IN_FEAT, {c0}::padded_independent_extent }}, 1, 1);',
            f'    num_buffers(g{gi}_buf_in) = 2;',
            f'    connect<>(g{gi}_ifm[0], g{gi}_buf_in.in[0]);',
            f'    write_access(g{gi}_buf_in.in[0]) = tiling({{',
            f'      .buffer_dimension = {{ {c0}::IN_FEAT, {c0}::padded_independent_extent }},',
            f'      .tiling_dimension = {{ {c0}::IN_FEAT, {c0}::padded_independent_extent }},',
            f'      .offset = {{ 0, 0 }}', f'    }});')
        buf_read_tiled(c0, f'g{gi}_buf_in.out[0]')
        add(f'    connect<>(g{gi}_buf_in.out[0], g{gi}_l0.in1[0]);', '')
        for li in range(n - 1):
            ci = f"A{gi}L{li}Cfg"; ci1 = f"A{gi}L{li+1}Cfg"
            add(f'    g{gi}_buf_mid{li} = shared_buffer<typename {ci}::result_t>::create(',
                f'      {{ {ci}::OUT_FEAT, {ci}::padded_independent_extent }}, 1, 1);',
                f'    num_buffers(g{gi}_buf_mid{li}) = 2;',
                f'    connect<>(g{gi}_l{li}.out1[0], g{gi}_buf_mid{li}.in[0]);')
            buf_write_tiled(ci,  f'g{gi}_buf_mid{li}.in[0]')
            buf_read_tiled(ci1, f'g{gi}_buf_mid{li}.out[0]')
            add(f'    connect<>(g{gi}_buf_mid{li}.out[0], g{gi}_l{li+1}.in1[0]);', '')
        add(f'    g{gi}_buf_out = shared_buffer<typename {cl}::result_t>::create(',
            f'      {{ {cl}::OUT_FEAT, {cl}::padded_independent_extent }}, 1, 1);',
            f'    num_buffers(g{gi}_buf_out) = 2;',
            f'    connect<>(g{gi}_l{n-1}.out1[0], g{gi}_buf_out.in[0]);')
        buf_write_tiled(cl, f'g{gi}_buf_out.in[0]')
        add(f'    read_access(g{gi}_buf_out.out[0]) = tiling({{',
            f'      .buffer_dimension = {{ {cl}::OUT_FEAT, {cl}::padded_independent_extent }},',
            f'      .tiling_dimension = {{ {cl}::OUT_FEAT, {cl}::padded_independent_extent }},',
            f'      .offset = {{ 0, 0 }},',
            f'      .boundary_dimension = {{ {cl}::OUT_FEAT, {cl}::padded_independent_extent }}',
            f'    }});',
            f'    connect<>(g{gi}_buf_out.out[0], g{gi}_ofm[0]);', '')
        for li in range(n):
            c = f"A{gi}L{li}Cfg"
            add(f'    for (int ch = 0; ch < {c}::CAS_NUM; ++ch) {{',
                f'      for (int col = 0; col < {c}::CAS_LENGTH; ++col)',
                f'        connect<>(g{gi}_wts{li}[ch * {c}::CAS_LENGTH + col], g{gi}_l{li}.wts[ch * {c}::CAS_LENGTH + col]);',
                f'      connect<>(g{gi}_bias{li}[ch], g{gi}_l{li}.bias[ch]);',
                f'    }}')
        add('')
        for li in range(n):
            add(f'    g{gi}_l{li}.place_graph(A{gi}L{li}Cfg::col_placement, A{gi}L{li}Cfg::row_placement);')
        add('')
    add('  }', '};', '')

    add('class dut_graph : public graph {', 'public:')
    for gi in range(len(aie_groups)):
        add(f'  input_plio  plio_g{gi}_in;', f'  output_plio plio_g{gi}_out;')
    add('')
    for gi, group in enumerate(aie_groups):
        for li in range(len(group)):
            c = f"A{gi}L{li}Cfg"
            add(f'  input_port g{gi}_wts{li}[{c}::CAS_NUM * {c}::CAS_LENGTH];',
                f'  input_port g{gi}_bias{li}[{c}::CAS_NUM];')
    add('', '  top_graph dut;', '', '  dut_graph() {')
    for gi in range(len(aie_groups)):
        add(f'    plio_g{gi}_in  = input_plio::create("g{gi}_in",  plio_128_bits, "data/g{gi}_ifm.txt");',
            f'    plio_g{gi}_out = output_plio::create("g{gi}_out", plio_128_bits, "data/g{gi}_ofm.txt");',
            f'    connect<>(plio_g{gi}_in.out[0],  dut.g{gi}_ifm[0]);',
            f'    connect<>(dut.g{gi}_ofm[0], plio_g{gi}_out.in[0]);', '')
        for li in range(len(aie_groups[gi])):
            c = f"A{gi}L{li}Cfg"
            add(f'    for (int ch = 0; ch < {c}::CAS_NUM; ++ch) {{',
                f'      for (int col = 0; col < {c}::CAS_LENGTH; ++col)',
                f'        connect<>(g{gi}_wts{li}[ch * {c}::CAS_LENGTH + col], dut.g{gi}_wts{li}[ch * {c}::CAS_LENGTH + col]);',
                f'      connect<>(g{gi}_bias{li}[ch], dut.g{gi}_bias{li}[ch]);',
                f'    }}')
        add('')
    add('  }', '};', '', 'dut_graph dut;', '')

    add('#if defined(__AIESIM__) || defined(__X86SIM__)', 'int main() {', '  dut.init();', '')
    for gi, group in enumerate(aie_groups):
        for li in range(len(group)):
            c = f"A{gi}L{li}Cfg"
            add(f'  for (int ch = 0; ch < {c}::CAS_NUM; ++ch) {{',
                f'    for (int col = 0; col < {c}::CAS_LENGTH; ++col) {{',
                f'      int idx = ch * {c}::CAS_LENGTH + col;',
                f'      dut.update(dut.g{gi}_wts{li}[idx], g{gi}_w{li}[ch][col], {c}::IN_FEAT_SLICE * {c}::OUT_FEAT_SLICE);',
                f'    }}',
                f'    dut.update(dut.g{gi}_bias{li}[ch], g{gi}_b{li}[ch], {c}::OUT_FEAT_SLICE);',
                f'  }}')
    add('', '#ifdef __AIESIM__')
    for gi in range(len(aie_groups)):
        add(f'  event::handle h{gi} = event::start_profiling(',
            f'    dut.plio_g{gi}_in, dut.plio_g{gi}_out,',
            f'    event::io_stream_start_difference_cycles);')
    add('#endif', '', '  dut.run(N_ITER);', '  dut.wait();', '', '#ifdef __AIESIM__')
    for gi in range(len(aie_groups)):
        add(f'  long long cyc{gi} = event::read_profiling(h{gi});',
            f'  event::stop_profiling(h{gi});')
    add('  std::system("mkdir -p aiesimulator_output/data");')
    for gi in range(len(aie_groups)):
        add(f'  {{ std::ofstream lf("aiesimulator_output/data/g{gi}_latency.json");',
            f'    lf << "{{\\\"cycles\\\": " << cyc{gi} << "}}\\n"; }}')
    add('#endif', '', '  dut.end();', '  return 0;', '}', '#endif')
    return "\n".join(L) + "\n"

def write_plio(path, x):
    buf = io.StringIO()
    np.savetxt(buf, x.flatten().astype(np.int32).reshape(-1, 16), fmt="%d")
    write_if_changed(path, buf.getvalue())

# ── PL code generation ────────────────────────────────────────────────────────

def pl_weight_header(name, W):
    return f"static const ap_int<8> {name}[] = {{{', '.join(str(int(v)) for v in W.flatten())}}};\n"

def pl_bias_header(name, b):
    return f"static const ap_int<32> {name}[] = {{{', '.join(str(int(v)) for v in b)}}};\n"

def pl_group_cpp(pi, group, batch, shift, reuse, prev_is_aie):
    n         = len(group)
    n_in_0    = group[0][1]
    n_out_n   = group[-1][2]
    in_beats  = batch * n_in_0  // 16
    out_beats = batch * n_out_n // 16
    in_t      = "ap_uint<8>" if prev_is_aie else "ap_int<8>"
    in_cast   = "(ap_uint<8>)" if prev_is_aie else "(ap_int<8>)"
    fn0       = "dense_relu_u8" if prev_is_aie else "dense_relu"

    L = ['// Generated by gen.py', '#include <ap_int.h>', '#include <hls_stream.h>',
         '#include <ap_axi_sdata.h>', '#include "dense.h"', '']
    for li in range(n):
        L += [f'#include "pl_weights/g{pi}_w{li}.h"', f'#include "pl_weights/g{pi}_b{li}.h"']
    L += ['', 'extern "C" {', f'void pl_group{pi}(',
          '    hls::stream<ap_axis<128,0,0,0>>& in_s,',
          '    hls::stream<ap_axis<128,0,0,0>>& out_s)', '{',
          '#pragma HLS INTERFACE axis port=in_s',
          '#pragma HLS INTERFACE axis port=out_s',
          '// Free-running kernel: processes data as it arrives.',
          '#pragma HLS INTERFACE ap_ctrl_none port=return', '',
          f'    {in_t} buf_in[{batch * n_in_0}];']
    for li in range(n - 1):
        L.append(f'    ap_int<8> mid{li}[{batch * group[li][2]}];')
    L += [f'    ap_int<8> buf_out[{batch * n_out_n}];', '',
          f'    for (int beat = 0; beat < {in_beats}; ++beat) {{',
          '#pragma HLS PIPELINE',
          '        ap_axis<128,0,0,0> w = in_s.read();',
          '        ap_uint<128> d = w.data;',
          '        for (int b = 0; b < 16; ++b)',
          f'            buf_in[beat*16+b] = {in_cast}(d >> (b*8));',
          '    }', '']
    for li, (_, ni, no) in enumerate(group):
        in_buf  = "buf_in"    if li == 0     else f"mid{li-1}"
        out_buf = "buf_out"   if li == n - 1 else f"mid{li}"
        fn = fn0 if li == 0 else "dense_relu"
        L += [f'    using Cfg{li} = nnet::dense_config<{ni}, {no}, {reuse}>;',
              f'    for (int s = 0; s < {batch}; ++s)',
              f'        nnet::{fn}<Cfg{li}>({in_buf}+s*{ni}, {out_buf}+s*{no},',
              f'            (ap_int<8>*)g{pi}_w{li}, (ap_int<32>*)g{pi}_b{li}, {shift});', '']
    L += [f'    for (int beat = 0; beat < {out_beats}; ++beat) {{',
          '#pragma HLS PIPELINE',
          '        ap_axis<128,0,0,0> w;',
          '        w.keep = -1; w.strb = -1;',
          f'        w.last = (beat == {out_beats}-1) ? 1 : 0;',
          '        ap_uint<128> d = 0;',
          '        for (int b = 0; b < 16; ++b)',
          f'            d |= ((ap_uint<128>)(ap_uint<8>)buf_out[beat*16+b]) << (b*8);',
          '        w.data = d;',
          '        out_s.write(w);',
          '    }', '}', '}']
    return "\n".join(L) + "\n"

def hls_cfg(src_abs, top, xo_abs, extra_cflags=""):
    cflags = f"\nsyn.cflags={extra_cflags}" if extra_cflags else ""
    return (f"[hls]\nflow_target=vitis\nsyn.file={src_abs}\nsyn.top={top}\n"
            f"package.ip.name={top}\npackage.output.syn=true\n"
            f"package.output.format=xo\npackage.output.file={xo_abs}{cflags}\n")

def system_cfg(segs):
    L = ['[connectivity]', 'nk=mm2s:1:mm2s', 'nk=s2mm:1:s2mm']
    for t, gi, _ in segs:
        if t == 'pl': L.append(f'nk=pl_group{gi}:1:pl_group{gi}')
    L.append('')
    prev = 'mm2s.s'
    for t, gi, _ in segs:
        if t == 'aie':
            L.append(f'sc={prev}:ai_engine_0.g{gi}_in'); prev = f'ai_engine_0.g{gi}_out'
        else:
            L.append(f'sc={prev}:pl_group{gi}.in_s');    prev = f'pl_group{gi}.out_s'
    L.append(f'sc={prev}:s2mm.s')
    return "\n".join(L) + "\n"

def sw_data_h(x0, ref_out):
    B, N_IN = x0.shape; N_OUT = ref_out.shape[1]
    def arr(name, dtype, vals):
        return f"static const {dtype} {name}[] = {{{', '.join(str(int(v)) for v in vals)}}};\n"
    return (f"#pragma once\n#include <cstdint>\n\n"
            f"static const int kBatch = {B};\n"
            f"static const int kNIn   = {N_IN};\n"
            f"static const int kNOut  = {N_OUT};\n\n"
            + arr("kInput",  "int8_t",  x0.flatten().astype(np.int8))
            + arr("kGolden", "uint8_t", ref_out.flatten().astype(np.uint8)))

def host_cpp(batch, n_in, n_out, aie_groups_info, n_iter=1):
    # aie_groups_info: [(gi, [(li, cas_num, cas_length, in_sl, out_sl), ...]), ...]
    in_bytes  = batch * n_in
    out_bytes = batch * n_out
    in_words  = (in_bytes  + 3) // 4
    out_words = (out_bytes + 3) // 4

    inc_lines, upd_lines = [], []
    for gi, layers in aie_groups_info:
        for li, cas_num, cas_length, in_sl, out_sl in layers:
            inc_lines += [f'#include "../aie/weights/g{gi}_w{li}.h"',
                          f'#include "../aie/weights/g{gi}_b{li}.h"']
            for ch in range(cas_num):
                for col in range(cas_length):
                    idx = ch * cas_length + col
                    upd_lines.append(
                        f'    graph.update("dut.g{gi}_wts{li}[{idx}]", '
                        f'(void*)g{gi}_w{li}[{ch}][{col}], {in_sl * out_sl});')
                upd_lines.append(
                    f'    graph.update("dut.g{gi}_bias{li}[{ch}]", '
                    f'(void*)g{gi}_b{li}[{ch}], {out_sl * 4});')
    wt_includes = "\n".join(inc_lines)
    wt_updates  = "\n".join(upd_lines)

    return f"""\
// Generated by gen.py
#include <chrono>
#include <cstring>
#include <iostream>
#include "experimental/xrt_kernel.h"
#include "experimental/xrt_graph.h"
#include "data.h"
{wt_includes}

int main(int argc, char* argv[]) {{
    if (argc < 2) {{ std::cerr << "Usage: host.exe <xclbin>\\n"; return 1; }}
    auto device = xrt::device(0);
    auto uuid   = device.load_xclbin(argv[1]);
    auto in_bo  = xrt::bo(device, {in_bytes},  xrt::bo::flags::normal, 0);
    auto out_bo = xrt::bo(device, {out_bytes}, xrt::bo::flags::normal, 0);
    std::memcpy(in_bo.map<void*>(), kInput, {in_bytes});
    in_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    auto graph  = xrt::graph(device, uuid, "dut");
    auto mm2s_k = xrt::kernel(device, uuid, "mm2s");
    auto s2mm_k = xrt::kernel(device, uuid, "s2mm");
    auto s2mm_r = s2mm_k(out_bo, nullptr, {out_words});
    auto mm2s_r = mm2s_k(in_bo,  nullptr, {in_words});
{wt_updates}
    auto t0 = std::chrono::high_resolution_clock::now();
    graph.run({n_iter});
    graph.wait();
    mm2s_r.wait(); s2mm_r.wait();
    auto t1 = std::chrono::high_resolution_clock::now();
    double e2e_ns = std::chrono::duration<double, std::nano>(t1 - t0).count();
    std::cout << "End-to-end (hw_emu): " << e2e_ns << " ns\\n";
    out_bo.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    auto* out = out_bo.map<uint8_t*>();
    int errs = 0;
    for (int i = 0; i < {out_bytes}; ++i)
        if (out[i] != kGolden[i]) errs++;
    if (errs) std::cout << "FAIL: " << errs << " mismatches\\n";
    else       std::cout << "PASS\\n";
    return errs ? 1 : 0;
}}
"""

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Generate all build artefacts.")
    p.add_argument("--batch",        type=int, default=8)
    p.add_argument("--shift",        type=int, default=0)
    p.add_argument("--n-iter",       type=int, default=1)
    p.add_argument("--cas-length",   type=int, default=1)
    p.add_argument("--cas-num",      type=int, default=1)
    p.add_argument("--tile-m",       type=int, default=4)
    p.add_argument("--tile-k",       type=int, default=8)
    p.add_argument("--tile-n",       type=int, default=8)
    p.add_argument("--reuse-factor", type=int, default=1)
    p.add_argument("--seed",         type=int, default=42)
    a = p.parse_args()

    for d in ("data", "aie/weights", "pl/pl_weights", "pl_kernels", "sw", "generated"):
        os.makedirs(d, exist_ok=True)

    rng = np.random.default_rng(a.seed)
    weights = {i: rng.integers(-3, 4, (ni, no), dtype=np.int8)  for i, (_, ni, no) in enumerate(LAYERS)}
    biases  = {i: np.zeros(no, dtype=np.int32)                   for i, (_, ni, no) in enumerate(LAYERS)}

    segs      = segments()
    aie_segs  = [(gi, idxs) for t, gi, idxs in segs if t == 'aie']
    pl_segs   = [(gi, idxs) for t, gi, idxs in segs if t == 'pl']
    aie_groups = [[LAYERS[i] for i in idxs] for _, idxs in aie_segs]

    x0 = np.random.default_rng(0).integers(0, 4, (a.batch, LAYERS[0][1]), dtype=np.int8)
    aie_inputs, aie_refs, ref_out = forward(x0, weights, biases, segs, a.shift)

    # AIE files
    write_if_changed("aie/parameters.h", parameters_h(aie_groups, a))
    write_if_changed("aie/graph.cpp",    graph_cpp(aie_groups, a))
    for gi, idxs in aie_segs:
        write_plio(f"data/g{gi}_ifm.txt", aie_inputs[gi])
        np.save(f"data/g{gi}_ref.npy", aie_refs[gi])
        for li, idx in enumerate(idxs):
            _, ni, no = LAYERS[idx]
            in_sl, out_sl = ni // a.cas_length, no // a.cas_num
            W = pack_weights(weights[idx], a.tile_k, a.tile_n)
            write_if_changed(f"aie/weights/g{gi}_w{li}.h",
                aie_weight_header(f"g{gi}_w{li}", W, a.cas_num, a.cas_length, in_sl, out_sl))
            write_if_changed(f"aie/weights/g{gi}_b{li}.h",
                aie_bias_header(f"g{gi}_b{li}", biases[idx], a.cas_num, out_sl))

    # PL / hw_emu files
    base = os.path.abspath(".")
    prev_types = {gi: (segs[si-1][0] if si > 0 else None)
                  for si, (t, gi, _) in enumerate(segs) if t == 'pl'}

    write_if_changed("pl_kernels/mm2s.cfg",
        hls_cfg(f"{base}/pl/mm2s.cpp", "mm2s", f"{base}/pl_kernels/mm2s.xo"))
    write_if_changed("pl_kernels/s2mm.cfg",
        hls_cfg(f"{base}/pl/s2mm.cpp", "s2mm", f"{base}/pl_kernels/s2mm.xo"))

    for pi, (gi, idxs) in enumerate(pl_segs):
        group = [LAYERS[i] for i in idxs]
        for li, idx in enumerate(idxs):
            write_if_changed(f"pl/pl_weights/g{pi}_w{li}.h", pl_weight_header(f"g{pi}_w{li}", weights[idx]))
            write_if_changed(f"pl/pl_weights/g{pi}_b{li}.h", pl_bias_header(f"g{pi}_b{li}", biases[idx]))
        write_if_changed(f"pl/pl_group{pi}.cpp",
            pl_group_cpp(pi, group, a.batch, a.shift, a.reuse_factor,
                         prev_is_aie=(prev_types.get(gi) == 'aie')))
        write_if_changed(f"pl_kernels/pl_group{pi}.cfg",
            hls_cfg(f"{base}/pl/pl_group{pi}.cpp", f"pl_group{pi}",
                    f"{base}/pl_kernels/pl_group{pi}.xo", extra_cflags=f"-I{base}/pl"))

    write_if_changed("system.cfg",   system_cfg(segs))
    aie_groups_info = []
    for gi, idxs in aie_segs:
        layers_info = []
        for li, idx in enumerate(idxs):
            _, ni, no = LAYERS[idx]
            layers_info.append((li, a.cas_num, a.cas_length, ni // a.cas_length, no // a.cas_num))
        aie_groups_info.append((gi, layers_info))
    write_if_changed("sw/data.h",    sw_data_h(x0, ref_out))
    write_if_changed("sw/host.cpp",  host_cpp(a.batch, LAYERS[0][1], LAYERS[-1][2], aie_groups_info, a.n_iter))

    _sh = "#!/bin/bash\nexport XILINX_XRT=/usr\n./host.exe a.xclbin\n"
    if not os.path.exists("sw/embedded_exec.sh") or open("sw/embedded_exec.sh").read() != _sh:
        with open("sw/embedded_exec.sh", "w") as f: f.write(_sh)
        os.chmod("sw/embedded_exec.sh", 0o755)

    # Tell Make which PL group XOs exist
    write_if_changed("generated/groups.mk",
        "PL_GROUP_XOS := " + " ".join(f"pl_kernels/pl_group{pi}.xo"
                                       for pi in range(len(pl_segs))) + "\n")
    # Config for check.py
    pl_cycles = sum(a.batch * LAYERS[idx][1]
                    for _, idxs in pl_segs for idx in idxs)
    json.dump({"batch": a.batch, "n_aie_groups": len(aie_segs), "pl_cycles_est": pl_cycles},
              open("data/config.json", "w"))

if __name__ == "__main__":
    main()
