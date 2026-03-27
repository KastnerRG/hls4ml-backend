#include <adf.h>
#include <fstream>
#include "parameters.h"
#include "dense_graph.h"

extern "C" {
  #include "weights/g0_w0.h"
  #include "weights/g0_b0.h"
}

using namespace adf;

class top_graph : public graph {
public:
  input_port  g0_ifm[1];
  output_port g0_ofm[1];

  input_port g0_wts0[A0L0Cfg::CAS_NUM * A0L0Cfg::CAS_LENGTH];
  input_port g0_bias0[A0L0Cfg::CAS_NUM];

private:
  dense_bias_relu_graph<A0L0Cfg> g0_l0;

  shared_buffer<typename A0L0Cfg::data_t>    g0_buf_in;
  shared_buffer<typename A0L0Cfg::result_t> g0_buf_out;

public:
  top_graph() {
    // ── Group 0 ──
    g0_buf_in = shared_buffer<typename A0L0Cfg::data_t>::create(
      { A0L0Cfg::IN_FEAT, A0L0Cfg::padded_independent_extent }, 1, 1);
    num_buffers(g0_buf_in) = 2;
    connect<>(g0_ifm[0], g0_buf_in.in[0]);
    write_access(g0_buf_in.in[0]) = tiling({
      .buffer_dimension = { A0L0Cfg::IN_FEAT, A0L0Cfg::padded_independent_extent },
      .tiling_dimension = { A0L0Cfg::IN_FEAT, A0L0Cfg::padded_independent_extent },
      .offset = { 0, 0 }
    });
    read_access(g0_buf_in.out[0]) = tiling({
      .buffer_dimension = { A0L0Cfg::IN_FEAT, A0L0Cfg::padded_independent_extent },
      .tiling_dimension = { A0L0Cfg::K, A0L0Cfg::M },
      .offset = { 0, 0 },
      .tile_traversal = {
        { .dimension = 0, .stride = A0L0Cfg::K, .wrap = A0L0Cfg::IN_FEAT / A0L0Cfg::K },
        { .dimension = 1, .stride = A0L0Cfg::M, .wrap = A0L0Cfg::padded_independent_extent / A0L0Cfg::M }
      },
      .boundary_dimension = { A0L0Cfg::IN_FEAT, A0L0Cfg::padded_independent_extent }
    });
    connect<>(g0_buf_in.out[0], g0_l0.in1[0]);

    g0_buf_out = shared_buffer<typename A0L0Cfg::result_t>::create(
      { A0L0Cfg::OUT_FEAT, A0L0Cfg::padded_independent_extent }, 1, 1);
    num_buffers(g0_buf_out) = 2;
    connect<>(g0_l0.out1[0], g0_buf_out.in[0]);
    write_access(g0_buf_out.in[0]) = tiling({
      .buffer_dimension = { A0L0Cfg::OUT_FEAT, A0L0Cfg::padded_independent_extent },
      .tiling_dimension = { A0L0Cfg::N, A0L0Cfg::M },
      .offset = { 0, 0 },
      .tile_traversal = {
        { .dimension = 0, .stride = A0L0Cfg::N, .wrap = A0L0Cfg::OUT_FEAT / A0L0Cfg::N },
        { .dimension = 1, .stride = A0L0Cfg::M, .wrap = A0L0Cfg::padded_independent_extent / A0L0Cfg::M }
      }
    });
    read_access(g0_buf_out.out[0]) = tiling({
      .buffer_dimension = { A0L0Cfg::OUT_FEAT, A0L0Cfg::padded_independent_extent },
      .tiling_dimension = { A0L0Cfg::OUT_FEAT, A0L0Cfg::padded_independent_extent },
      .offset = { 0, 0 },
      .boundary_dimension = { A0L0Cfg::OUT_FEAT, A0L0Cfg::padded_independent_extent }
    });
    connect<>(g0_buf_out.out[0], g0_ofm[0]);

    for (int ch = 0; ch < A0L0Cfg::CAS_NUM; ++ch) {
      for (int col = 0; col < A0L0Cfg::CAS_LENGTH; ++col)
        connect<>(g0_wts0[ch * A0L0Cfg::CAS_LENGTH + col], g0_l0.wts[ch * A0L0Cfg::CAS_LENGTH + col]);
      connect<>(g0_bias0[ch], g0_l0.bias[ch]);
    }

    g0_l0.place_graph(A0L0Cfg::col_placement, A0L0Cfg::row_placement);

  }
};

class dut_graph : public graph {
public:
  input_plio  plio_g0_in;
  output_plio plio_g0_out;

  input_port g0_wts0[A0L0Cfg::CAS_NUM * A0L0Cfg::CAS_LENGTH];
  input_port g0_bias0[A0L0Cfg::CAS_NUM];

  top_graph dut;

  dut_graph() {
    plio_g0_in  = input_plio::create("g0_in",  plio_128_bits, "data/g0_ifm.txt");
    plio_g0_out = output_plio::create("g0_out", plio_128_bits, "data/g0_ofm.txt");
    connect<>(plio_g0_in.out[0],  dut.g0_ifm[0]);
    connect<>(dut.g0_ofm[0], plio_g0_out.in[0]);

    for (int ch = 0; ch < A0L0Cfg::CAS_NUM; ++ch) {
      for (int col = 0; col < A0L0Cfg::CAS_LENGTH; ++col)
        connect<>(g0_wts0[ch * A0L0Cfg::CAS_LENGTH + col], dut.g0_wts0[ch * A0L0Cfg::CAS_LENGTH + col]);
      connect<>(g0_bias0[ch], dut.g0_bias0[ch]);
    }

  }
};

dut_graph dut;

#if defined(__AIESIM__) || defined(__X86SIM__)
int main() {
  dut.init();

  for (int ch = 0; ch < A0L0Cfg::CAS_NUM; ++ch) {
    for (int col = 0; col < A0L0Cfg::CAS_LENGTH; ++col) {
      int idx = ch * A0L0Cfg::CAS_LENGTH + col;
      dut.update(dut.g0_wts0[idx], g0_w0[ch][col], A0L0Cfg::IN_FEAT_SLICE * A0L0Cfg::OUT_FEAT_SLICE);
    }
    dut.update(dut.g0_bias0[ch], g0_b0[ch], A0L0Cfg::OUT_FEAT_SLICE);
  }

#ifdef __AIESIM__
  event::handle h0 = event::start_profiling(
    dut.plio_g0_in, dut.plio_g0_out,
    event::io_stream_start_difference_cycles);
#endif

  dut.run(N_ITER);
  dut.wait();

#ifdef __AIESIM__
  long long cyc0 = event::read_profiling(h0);
  event::stop_profiling(h0);
  std::system("mkdir -p aiesimulator_output/data");
  { std::ofstream lf("aiesimulator_output/data/g0_latency.json");
    lf << "{\"cycles\": " << cyc0 << "}\n"; }
#endif

  dut.end();
  return 0;
}
#endif
