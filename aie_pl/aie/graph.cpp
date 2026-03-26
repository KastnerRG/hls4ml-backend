#include <adf.h>
#include <fstream>
#include "parameters.h"
#include "dense_graph.h"

extern "C" {
  #include "weights/weights_l3.h"
  #include "weights/bias_l3.h"
  #include "weights/weights_l4.h"
  #include "weights/bias_l4.h"
}

using namespace adf;

// ── Top-level graph: 2 AIE dense layers (L3, L4) connected via MemTile buffers ──

class top_graph : public graph {
public:
  input_port  ifm[1];
  output_port ofm[1];

  input_port wts3 [L3Cfg::CAS_NUM * L3Cfg::CAS_LENGTH];
  input_port bias3[L3Cfg::CAS_NUM];
  input_port wts4 [L4Cfg::CAS_NUM * L4Cfg::CAS_LENGTH];
  input_port bias4[L4Cfg::CAS_NUM];

private:
  dense_bias_relu_graph<L3Cfg> l3;
  dense_bias_relu_graph<L4Cfg> l4;

  shared_buffer<typename L3Cfg::data_t>   buffer_in;
  shared_buffer<typename L3Cfg::result_t> buffer_mid;
  shared_buffer<typename L4Cfg::result_t> buffer_out;

public:
  top_graph() {
    // ── Input buffer: PLIO → L3 ──
    // dim: {FEAT, BATCH}, write full (row-major from PLIO), read as (K,M) tiles
    buffer_in = shared_buffer<typename L3Cfg::data_t>::create(
      { L3Cfg::IN_FEAT, L3Cfg::padded_independent_extent }, 1, 1);
    num_buffers(buffer_in) = 2;

    connect<>(ifm[0], buffer_in.in[0]);
    write_access(buffer_in.in[0]) = tiling({
      .buffer_dimension = { L3Cfg::IN_FEAT, L3Cfg::padded_independent_extent },
      .tiling_dimension = { L3Cfg::IN_FEAT, L3Cfg::padded_independent_extent },
      .offset           = { 0, 0 }
    });
    read_access(buffer_in.out[0]) = tiling({
      .buffer_dimension = { L3Cfg::IN_FEAT, L3Cfg::padded_independent_extent },
      .tiling_dimension = { L3Cfg::K, L3Cfg::M },
      .offset           = { 0, 0 },
      .tile_traversal   = {
        { .dimension = 0, .stride = L3Cfg::K, .wrap = L3Cfg::IN_FEAT  / L3Cfg::K },
        { .dimension = 1, .stride = L3Cfg::M, .wrap = L3Cfg::padded_independent_extent / L3Cfg::M }
      },
      .boundary_dimension = { L3Cfg::IN_FEAT, L3Cfg::padded_independent_extent }
    });
    connect<>(buffer_in.out[0], l3.in1[0]);

    // ── Intermediate buffer: L3 → L4 ──
    buffer_mid = shared_buffer<typename L3Cfg::result_t>::create(
      { L3Cfg::OUT_FEAT, L3Cfg::padded_independent_extent }, 1, 1);
    num_buffers(buffer_mid) = 2;

    connect<>(l3.out1[0], buffer_mid.in[0]);
    write_access(buffer_mid.in[0]) = tiling({
      .buffer_dimension = { L3Cfg::OUT_FEAT, L3Cfg::padded_independent_extent },
      .tiling_dimension = { L3Cfg::N, L3Cfg::M },
      .offset           = { 0, 0 },
      .tile_traversal   = {
        { .dimension = 0, .stride = L3Cfg::N, .wrap = L3Cfg::OUT_FEAT / L3Cfg::N },
        { .dimension = 1, .stride = L3Cfg::M, .wrap = L3Cfg::padded_independent_extent / L3Cfg::M }
      }
    });
    read_access(buffer_mid.out[0]) = tiling({
      .buffer_dimension = { L4Cfg::IN_FEAT, L4Cfg::padded_independent_extent },
      .tiling_dimension = { L4Cfg::K, L4Cfg::M },
      .offset           = { 0, 0 },
      .tile_traversal   = {
        { .dimension = 0, .stride = L4Cfg::K, .wrap = L4Cfg::IN_FEAT / L4Cfg::K },
        { .dimension = 1, .stride = L4Cfg::M, .wrap = L4Cfg::padded_independent_extent / L4Cfg::M }
      },
      .boundary_dimension = { L4Cfg::IN_FEAT, L4Cfg::padded_independent_extent }
    });
    connect<>(buffer_mid.out[0], l4.in1[0]);

    // ── Output buffer: L4 → PLIO ──
    buffer_out = shared_buffer<typename L4Cfg::result_t>::create(
      { L4Cfg::OUT_FEAT, L4Cfg::padded_independent_extent }, 1, 1);
    num_buffers(buffer_out) = 2;

    connect<>(l4.out1[0], buffer_out.in[0]);
    write_access(buffer_out.in[0]) = tiling({
      .buffer_dimension = { L4Cfg::OUT_FEAT, L4Cfg::padded_independent_extent },
      .tiling_dimension = { L4Cfg::N, L4Cfg::M },
      .offset           = { 0, 0 },
      .tile_traversal   = {
        { .dimension = 0, .stride = L4Cfg::N, .wrap = L4Cfg::OUT_FEAT / L4Cfg::N },
        { .dimension = 1, .stride = L4Cfg::M, .wrap = L4Cfg::padded_independent_extent / L4Cfg::M }
      }
    });
    read_access(buffer_out.out[0]) = tiling({
      .buffer_dimension = { L4Cfg::OUT_FEAT, L4Cfg::padded_independent_extent },
      .tiling_dimension = { L4Cfg::OUT_FEAT, L4Cfg::padded_independent_extent },
      .offset           = { 0, 0 },
      .boundary_dimension = { L4Cfg::OUT_FEAT, L4Cfg::padded_independent_extent }
    });
    connect<>(buffer_out.out[0], ofm[0]);

    // ── Weight / bias connections ──
    for (int ch = 0; ch < L3Cfg::CAS_NUM; ++ch) {
      for (int col = 0; col < L3Cfg::CAS_LENGTH; ++col)
        connect<>(wts3[ch * L3Cfg::CAS_LENGTH + col], l3.wts[ch * L3Cfg::CAS_LENGTH + col]);
      connect<>(bias3[ch], l3.bias[ch]);
    }
    for (int ch = 0; ch < L4Cfg::CAS_NUM; ++ch) {
      for (int col = 0; col < L4Cfg::CAS_LENGTH; ++col)
        connect<>(wts4[ch * L4Cfg::CAS_LENGTH + col], l4.wts[ch * L4Cfg::CAS_LENGTH + col]);
      connect<>(bias4[ch], l4.bias[ch]);
    }

    l3.place_graph(L3Cfg::col_placement, L3Cfg::row_placement);
    l4.place_graph(L4Cfg::col_placement, L4Cfg::row_placement);
  }
};


// ── DUT: PLIO wrappers around top_graph ──

class dut_graph : public graph {
public:
  input_plio  plio_in;
  output_plio plio_out;

  input_port wts3 [L3Cfg::CAS_NUM * L3Cfg::CAS_LENGTH];
  input_port bias3[L3Cfg::CAS_NUM];
  input_port wts4 [L4Cfg::CAS_NUM * L4Cfg::CAS_LENGTH];
  input_port bias4[L4Cfg::CAS_NUM];

  top_graph dut;

  dut_graph() {
    plio_in  = input_plio::create("PLIO_in",  plio_128_bits, "data/ifm.txt");
    plio_out = output_plio::create("PLIO_out", plio_128_bits, "data/out.txt");

    connect<>(plio_in.out[0], dut.ifm[0]);
    connect<>(dut.ofm[0], plio_out.in[0]);

    for (int ch = 0; ch < L3Cfg::CAS_NUM; ++ch) {
      for (int col = 0; col < L3Cfg::CAS_LENGTH; ++col)
        connect<>(wts3[ch * L3Cfg::CAS_LENGTH + col], dut.wts3[ch * L3Cfg::CAS_LENGTH + col]);
      connect<>(bias3[ch], dut.bias3[ch]);
    }
    for (int ch = 0; ch < L4Cfg::CAS_NUM; ++ch) {
      for (int col = 0; col < L4Cfg::CAS_LENGTH; ++col)
        connect<>(wts4[ch * L4Cfg::CAS_LENGTH + col], dut.wts4[ch * L4Cfg::CAS_LENGTH + col]);
      connect<>(bias4[ch], dut.bias4[ch]);
    }
  }
};

dut_graph dut;

#if defined(__AIESIM__) || defined(__X86SIM__)
int main() {
  dut.init();

  for (int ch = 0; ch < L3Cfg::CAS_NUM; ++ch) {
    for (int col = 0; col < L3Cfg::CAS_LENGTH; ++col) {
      int idx = ch * L3Cfg::CAS_LENGTH + col;
      dut.update(dut.wts3[idx], weights_l3[ch][col], L3Cfg::IN_FEAT_SLICE * L3Cfg::OUT_FEAT_SLICE);
    }
    dut.update(dut.bias3[ch], bias_l3[ch], L3Cfg::OUT_FEAT_SLICE);
  }
  for (int ch = 0; ch < L4Cfg::CAS_NUM; ++ch) {
    for (int col = 0; col < L4Cfg::CAS_LENGTH; ++col) {
      int idx = ch * L4Cfg::CAS_LENGTH + col;
      dut.update(dut.wts4[idx], weights_l4[ch][col], L4Cfg::IN_FEAT_SLICE * L4Cfg::OUT_FEAT_SLICE);
    }
    dut.update(dut.bias4[ch], bias_l4[ch], L4Cfg::OUT_FEAT_SLICE);
  }

#ifdef __AIESIM__
  event::handle h = event::start_profiling(
    dut.plio_in, dut.plio_out, event::io_stream_start_difference_cycles);
#endif

  dut.run(N_ITER);
  dut.wait();

#ifdef __AIESIM__
  long long cycles = event::read_profiling(h);
  event::stop_profiling(h);
  std::system("mkdir -p aiesimulator_output/data");
  std::ofstream lf("aiesimulator_output/data/latency.json");
  lf << "{\"cycles\": " << cycles << "}\n";
#endif

  dut.end();
  return 0;
}
#endif
