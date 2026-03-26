#include <adf.h>
#include <fstream>
#include "parameters.h"
#include "dense_graph.h"

extern "C" {
  #include "weights/weights_a0.h"
  #include "weights/bias_a0.h"
  #include "weights/weights_a1.h"
  #include "weights/bias_a1.h"
}

using namespace adf;

class top_graph : public graph {
public:
  input_port  ifm[1];
  output_port ofm[1];

  input_port wts0 [A0Cfg::CAS_NUM * A0Cfg::CAS_LENGTH];
  input_port bias0[A0Cfg::CAS_NUM];
  input_port wts1 [A1Cfg::CAS_NUM * A1Cfg::CAS_LENGTH];
  input_port bias1[A1Cfg::CAS_NUM];

private:
  dense_bias_relu_graph<A0Cfg> l0;
  dense_bias_relu_graph<A1Cfg> l1;

  shared_buffer<typename A0Cfg::data_t>    buffer_in;
  shared_buffer<typename A0Cfg::result_t>  buffer_mid0;
  shared_buffer<typename A1Cfg::result_t> buffer_out;

public:
  top_graph() {
    buffer_in = shared_buffer<typename A0Cfg::data_t>::create(
      { A0Cfg::IN_FEAT, A0Cfg::padded_independent_extent }, 1, 1);
    num_buffers(buffer_in) = 2;
    connect<>(ifm[0], buffer_in.in[0]);
    write_access(buffer_in.in[0]) = tiling({
      .buffer_dimension = { A0Cfg::IN_FEAT, A0Cfg::padded_independent_extent },
      .tiling_dimension = { A0Cfg::IN_FEAT, A0Cfg::padded_independent_extent },
      .offset = { 0, 0 }
    });
    read_access(buffer_in.out[0]) = tiling({
      .buffer_dimension = { A0Cfg::IN_FEAT, A0Cfg::padded_independent_extent },
      .tiling_dimension = { A0Cfg::K, A0Cfg::M },
      .offset = { 0, 0 },
      .tile_traversal = {
        { .dimension = 0, .stride = A0Cfg::K, .wrap = A0Cfg::IN_FEAT / A0Cfg::K },
        { .dimension = 1, .stride = A0Cfg::M, .wrap = A0Cfg::padded_independent_extent / A0Cfg::M }
      },
      .boundary_dimension = { A0Cfg::IN_FEAT, A0Cfg::padded_independent_extent }
    });
    connect<>(buffer_in.out[0], l0.in1[0]);

    buffer_mid0 = shared_buffer<typename A0Cfg::result_t>::create(
      { A0Cfg::OUT_FEAT, A0Cfg::padded_independent_extent }, 1, 1);
    num_buffers(buffer_mid0) = 2;
    connect<>(l0.out1[0], buffer_mid0.in[0]);
    write_access(buffer_mid0.in[0]) = tiling({
      .buffer_dimension = { A0Cfg::OUT_FEAT, A0Cfg::padded_independent_extent },
      .tiling_dimension = { A0Cfg::N, A0Cfg::M },
      .offset = { 0, 0 },
      .tile_traversal = {
        { .dimension = 0, .stride = A0Cfg::N, .wrap = A0Cfg::OUT_FEAT / A0Cfg::N },
        { .dimension = 1, .stride = A0Cfg::M, .wrap = A0Cfg::padded_independent_extent / A0Cfg::M }
      }
    });
    read_access(buffer_mid0.out[0]) = tiling({
      .buffer_dimension = { A1Cfg::IN_FEAT, A1Cfg::padded_independent_extent },
      .tiling_dimension = { A1Cfg::K, A1Cfg::M },
      .offset = { 0, 0 },
      .tile_traversal = {
        { .dimension = 0, .stride = A1Cfg::K, .wrap = A1Cfg::IN_FEAT / A1Cfg::K },
        { .dimension = 1, .stride = A1Cfg::M, .wrap = A1Cfg::padded_independent_extent / A1Cfg::M }
      },
      .boundary_dimension = { A1Cfg::IN_FEAT, A1Cfg::padded_independent_extent }
    });
    connect<>(buffer_mid0.out[0], l1.in1[0]);

    buffer_out = shared_buffer<typename A1Cfg::result_t>::create(
      { A1Cfg::OUT_FEAT, A1Cfg::padded_independent_extent }, 1, 1);
    num_buffers(buffer_out) = 2;
    connect<>(l1.out1[0], buffer_out.in[0]);
    write_access(buffer_out.in[0]) = tiling({
      .buffer_dimension = { A1Cfg::OUT_FEAT, A1Cfg::padded_independent_extent },
      .tiling_dimension = { A1Cfg::N, A1Cfg::M },
      .offset = { 0, 0 },
      .tile_traversal = {
        { .dimension = 0, .stride = A1Cfg::N, .wrap = A1Cfg::OUT_FEAT / A1Cfg::N },
        { .dimension = 1, .stride = A1Cfg::M, .wrap = A1Cfg::padded_independent_extent / A1Cfg::M }
      }
    });
    read_access(buffer_out.out[0]) = tiling({
      .buffer_dimension = { A1Cfg::OUT_FEAT, A1Cfg::padded_independent_extent },
      .tiling_dimension = { A1Cfg::OUT_FEAT, A1Cfg::padded_independent_extent },
      .offset = { 0, 0 },
      .boundary_dimension = { A1Cfg::OUT_FEAT, A1Cfg::padded_independent_extent }
    });
    connect<>(buffer_out.out[0], ofm[0]);

    for (int ch = 0; ch < A0Cfg::CAS_NUM; ++ch) {
      for (int col = 0; col < A0Cfg::CAS_LENGTH; ++col)
        connect<>(wts0[ch * A0Cfg::CAS_LENGTH + col], l0.wts[ch * A0Cfg::CAS_LENGTH + col]);
      connect<>(bias0[ch], l0.bias[ch]);
    }
    for (int ch = 0; ch < A1Cfg::CAS_NUM; ++ch) {
      for (int col = 0; col < A1Cfg::CAS_LENGTH; ++col)
        connect<>(wts1[ch * A1Cfg::CAS_LENGTH + col], l1.wts[ch * A1Cfg::CAS_LENGTH + col]);
      connect<>(bias1[ch], l1.bias[ch]);
    }

    l0.place_graph(A0Cfg::col_placement, A0Cfg::row_placement);
    l1.place_graph(A1Cfg::col_placement, A1Cfg::row_placement);
  }
};

// ── DUT: PLIO wrappers ──

class dut_graph : public graph {
public:
  input_plio  plio_in;
  output_plio plio_out;

  input_port wts0 [A0Cfg::CAS_NUM * A0Cfg::CAS_LENGTH];
  input_port bias0[A0Cfg::CAS_NUM];
  input_port wts1 [A1Cfg::CAS_NUM * A1Cfg::CAS_LENGTH];
  input_port bias1[A1Cfg::CAS_NUM];

  top_graph dut;

  dut_graph() {
    plio_in  = input_plio::create("PLIO_in",  plio_128_bits, "data/ifm.txt");
    plio_out = output_plio::create("PLIO_out", plio_128_bits, "data/out.txt");

    connect<>(plio_in.out[0], dut.ifm[0]);
    connect<>(dut.ofm[0], plio_out.in[0]);

    for (int ch = 0; ch < A0Cfg::CAS_NUM; ++ch) {
      for (int col = 0; col < A0Cfg::CAS_LENGTH; ++col)
        connect<>(wts0[ch * A0Cfg::CAS_LENGTH + col], dut.wts0[ch * A0Cfg::CAS_LENGTH + col]);
      connect<>(bias0[ch], dut.bias0[ch]);
    }
    for (int ch = 0; ch < A1Cfg::CAS_NUM; ++ch) {
      for (int col = 0; col < A1Cfg::CAS_LENGTH; ++col)
        connect<>(wts1[ch * A1Cfg::CAS_LENGTH + col], dut.wts1[ch * A1Cfg::CAS_LENGTH + col]);
      connect<>(bias1[ch], dut.bias1[ch]);
    }
  }
};

dut_graph dut;

#if defined(__AIESIM__) || defined(__X86SIM__)
int main() {
  dut.init();

  for (int ch = 0; ch < A0Cfg::CAS_NUM; ++ch) {
    for (int col = 0; col < A0Cfg::CAS_LENGTH; ++col) {
      int idx = ch * A0Cfg::CAS_LENGTH + col;
      dut.update(dut.wts0[idx], weights_a0[ch][col], A0Cfg::IN_FEAT_SLICE * A0Cfg::OUT_FEAT_SLICE);
    }
    dut.update(dut.bias0[ch], bias_a0[ch], A0Cfg::OUT_FEAT_SLICE);
  }
  for (int ch = 0; ch < A1Cfg::CAS_NUM; ++ch) {
    for (int col = 0; col < A1Cfg::CAS_LENGTH; ++col) {
      int idx = ch * A1Cfg::CAS_LENGTH + col;
      dut.update(dut.wts1[idx], weights_a1[ch][col], A1Cfg::IN_FEAT_SLICE * A1Cfg::OUT_FEAT_SLICE);
    }
    dut.update(dut.bias1[ch], bias_a1[ch], A1Cfg::OUT_FEAT_SLICE);
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
