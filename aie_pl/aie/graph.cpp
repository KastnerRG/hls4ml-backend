#include <adf.h>
#include <fstream>
#include "parameters.h"
#include "dense_graph.h"

extern "C" {
  #include "weights/g0_w0.h"
  #include "weights/g0_b0.h"
  #include "weights/g0_w1.h"
  #include "weights/g0_b1.h"
  #include "weights/g1_w0.h"
  #include "weights/g1_b0.h"
  #include "weights/g2_w0.h"
  #include "weights/g2_b0.h"
  #include "weights/g3_w0.h"
  #include "weights/g3_b0.h"
  #include "weights/g4_w0.h"
  #include "weights/g4_b0.h"
  #include "weights/g5_w0.h"
  #include "weights/g5_b0.h"
  #include "weights/g6_w0.h"
  #include "weights/g6_b0.h"
}

using namespace adf;

class top_graph : public graph {
public:
  input_port  g0_ifm[1];
  output_port g0_ofm[1];
  input_port  g1_ifm[1];
  output_port g1_ofm[1];
  input_port  g2_ifm[1];
  output_port g2_ofm[1];
  input_port  g3_ifm[1];
  output_port g3_ofm[1];
  input_port  g4_ifm[1];
  output_port g4_ofm[1];
  input_port  g5_ifm[1];
  output_port g5_ofm[1];
  input_port  g6_ifm[1];
  output_port g6_ofm[1];

  input_port g0_wts0[A0L0Cfg::CAS_NUM * A0L0Cfg::CAS_LENGTH];
  input_port g0_bias0[A0L0Cfg::CAS_NUM];
  input_port g0_wts1[A0L1Cfg::CAS_NUM * A0L1Cfg::CAS_LENGTH];
  input_port g0_bias1[A0L1Cfg::CAS_NUM];
  input_port g1_wts0[A1L0Cfg::CAS_NUM * A1L0Cfg::CAS_LENGTH];
  input_port g1_bias0[A1L0Cfg::CAS_NUM];
  input_port g2_wts0[A2L0Cfg::CAS_NUM * A2L0Cfg::CAS_LENGTH];
  input_port g2_bias0[A2L0Cfg::CAS_NUM];
  input_port g3_wts0[A3L0Cfg::CAS_NUM * A3L0Cfg::CAS_LENGTH];
  input_port g3_bias0[A3L0Cfg::CAS_NUM];
  input_port g4_wts0[A4L0Cfg::CAS_NUM * A4L0Cfg::CAS_LENGTH];
  input_port g4_bias0[A4L0Cfg::CAS_NUM];
  input_port g5_wts0[A5L0Cfg::CAS_NUM * A5L0Cfg::CAS_LENGTH];
  input_port g5_bias0[A5L0Cfg::CAS_NUM];
  input_port g6_wts0[A6L0Cfg::CAS_NUM * A6L0Cfg::CAS_LENGTH];
  input_port g6_bias0[A6L0Cfg::CAS_NUM];

private:
  dense_bias_relu_graph<A0L0Cfg> g0_l0;
  dense_bias_relu_graph<A0L1Cfg> g0_l1;
  dense_bias_relu_graph<A1L0Cfg> g1_l0;
  dense_bias_relu_graph<A2L0Cfg> g2_l0;
  dense_bias_relu_graph<A3L0Cfg> g3_l0;
  dense_bias_relu_graph<A4L0Cfg> g4_l0;
  dense_bias_relu_graph<A5L0Cfg> g5_l0;
  dense_bias_relu_graph<A6L0Cfg> g6_l0;

  shared_buffer<typename A0L0Cfg::data_t>    g0_buf_in;
  shared_buffer<typename A0L0Cfg::result_t>  g0_buf_mid0;
  shared_buffer<typename A0L1Cfg::result_t> g0_buf_out;
  shared_buffer<typename A1L0Cfg::data_t>    g1_buf_in;
  shared_buffer<typename A1L0Cfg::result_t> g1_buf_out;
  shared_buffer<typename A2L0Cfg::data_t>    g2_buf_in;
  shared_buffer<typename A2L0Cfg::result_t> g2_buf_out;
  shared_buffer<typename A3L0Cfg::data_t>    g3_buf_in;
  shared_buffer<typename A3L0Cfg::result_t> g3_buf_out;
  shared_buffer<typename A4L0Cfg::data_t>    g4_buf_in;
  shared_buffer<typename A4L0Cfg::result_t> g4_buf_out;
  shared_buffer<typename A5L0Cfg::data_t>    g5_buf_in;
  shared_buffer<typename A5L0Cfg::result_t> g5_buf_out;
  shared_buffer<typename A6L0Cfg::data_t>    g6_buf_in;
  shared_buffer<typename A6L0Cfg::result_t> g6_buf_out;

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

    g0_buf_mid0 = shared_buffer<typename A0L0Cfg::result_t>::create(
      { A0L0Cfg::OUT_FEAT, A0L0Cfg::padded_independent_extent }, 1, 1);
    num_buffers(g0_buf_mid0) = 2;
    connect<>(g0_l0.out1[0], g0_buf_mid0.in[0]);
    write_access(g0_buf_mid0.in[0]) = tiling({
      .buffer_dimension = { A0L0Cfg::OUT_FEAT, A0L0Cfg::padded_independent_extent },
      .tiling_dimension = { A0L0Cfg::N, A0L0Cfg::M },
      .offset = { 0, 0 },
      .tile_traversal = {
        { .dimension = 0, .stride = A0L0Cfg::N, .wrap = A0L0Cfg::OUT_FEAT / A0L0Cfg::N },
        { .dimension = 1, .stride = A0L0Cfg::M, .wrap = A0L0Cfg::padded_independent_extent / A0L0Cfg::M }
      }
    });
    read_access(g0_buf_mid0.out[0]) = tiling({
      .buffer_dimension = { A0L1Cfg::IN_FEAT, A0L1Cfg::padded_independent_extent },
      .tiling_dimension = { A0L1Cfg::K, A0L1Cfg::M },
      .offset = { 0, 0 },
      .tile_traversal = {
        { .dimension = 0, .stride = A0L1Cfg::K, .wrap = A0L1Cfg::IN_FEAT / A0L1Cfg::K },
        { .dimension = 1, .stride = A0L1Cfg::M, .wrap = A0L1Cfg::padded_independent_extent / A0L1Cfg::M }
      },
      .boundary_dimension = { A0L1Cfg::IN_FEAT, A0L1Cfg::padded_independent_extent }
    });
    connect<>(g0_buf_mid0.out[0], g0_l1.in1[0]);

    g0_buf_out = shared_buffer<typename A0L1Cfg::result_t>::create(
      { A0L1Cfg::OUT_FEAT, A0L1Cfg::padded_independent_extent }, 1, 1);
    num_buffers(g0_buf_out) = 2;
    connect<>(g0_l1.out1[0], g0_buf_out.in[0]);
    write_access(g0_buf_out.in[0]) = tiling({
      .buffer_dimension = { A0L1Cfg::OUT_FEAT, A0L1Cfg::padded_independent_extent },
      .tiling_dimension = { A0L1Cfg::N, A0L1Cfg::M },
      .offset = { 0, 0 },
      .tile_traversal = {
        { .dimension = 0, .stride = A0L1Cfg::N, .wrap = A0L1Cfg::OUT_FEAT / A0L1Cfg::N },
        { .dimension = 1, .stride = A0L1Cfg::M, .wrap = A0L1Cfg::padded_independent_extent / A0L1Cfg::M }
      }
    });
    read_access(g0_buf_out.out[0]) = tiling({
      .buffer_dimension = { A0L1Cfg::OUT_FEAT, A0L1Cfg::padded_independent_extent },
      .tiling_dimension = { A0L1Cfg::OUT_FEAT, A0L1Cfg::padded_independent_extent },
      .offset = { 0, 0 },
      .boundary_dimension = { A0L1Cfg::OUT_FEAT, A0L1Cfg::padded_independent_extent }
    });
    connect<>(g0_buf_out.out[0], g0_ofm[0]);

    for (int ch = 0; ch < A0L0Cfg::CAS_NUM; ++ch) {
      for (int col = 0; col < A0L0Cfg::CAS_LENGTH; ++col)
        connect<>(g0_wts0[ch * A0L0Cfg::CAS_LENGTH + col], g0_l0.wts[ch * A0L0Cfg::CAS_LENGTH + col]);
      connect<>(g0_bias0[ch], g0_l0.bias[ch]);
    }
    for (int ch = 0; ch < A0L1Cfg::CAS_NUM; ++ch) {
      for (int col = 0; col < A0L1Cfg::CAS_LENGTH; ++col)
        connect<>(g0_wts1[ch * A0L1Cfg::CAS_LENGTH + col], g0_l1.wts[ch * A0L1Cfg::CAS_LENGTH + col]);
      connect<>(g0_bias1[ch], g0_l1.bias[ch]);
    }

    g0_l0.place_graph(A0L0Cfg::col_placement, A0L0Cfg::row_placement);
    g0_l1.place_graph(A0L1Cfg::col_placement, A0L1Cfg::row_placement);

    // ── Group 1 ──
    g1_buf_in = shared_buffer<typename A1L0Cfg::data_t>::create(
      { A1L0Cfg::IN_FEAT, A1L0Cfg::padded_independent_extent }, 1, 1);
    num_buffers(g1_buf_in) = 2;
    connect<>(g1_ifm[0], g1_buf_in.in[0]);
    write_access(g1_buf_in.in[0]) = tiling({
      .buffer_dimension = { A1L0Cfg::IN_FEAT, A1L0Cfg::padded_independent_extent },
      .tiling_dimension = { A1L0Cfg::IN_FEAT, A1L0Cfg::padded_independent_extent },
      .offset = { 0, 0 }
    });
    read_access(g1_buf_in.out[0]) = tiling({
      .buffer_dimension = { A1L0Cfg::IN_FEAT, A1L0Cfg::padded_independent_extent },
      .tiling_dimension = { A1L0Cfg::K, A1L0Cfg::M },
      .offset = { 0, 0 },
      .tile_traversal = {
        { .dimension = 0, .stride = A1L0Cfg::K, .wrap = A1L0Cfg::IN_FEAT / A1L0Cfg::K },
        { .dimension = 1, .stride = A1L0Cfg::M, .wrap = A1L0Cfg::padded_independent_extent / A1L0Cfg::M }
      },
      .boundary_dimension = { A1L0Cfg::IN_FEAT, A1L0Cfg::padded_independent_extent }
    });
    connect<>(g1_buf_in.out[0], g1_l0.in1[0]);

    g1_buf_out = shared_buffer<typename A1L0Cfg::result_t>::create(
      { A1L0Cfg::OUT_FEAT, A1L0Cfg::padded_independent_extent }, 1, 1);
    num_buffers(g1_buf_out) = 2;
    connect<>(g1_l0.out1[0], g1_buf_out.in[0]);
    write_access(g1_buf_out.in[0]) = tiling({
      .buffer_dimension = { A1L0Cfg::OUT_FEAT, A1L0Cfg::padded_independent_extent },
      .tiling_dimension = { A1L0Cfg::N, A1L0Cfg::M },
      .offset = { 0, 0 },
      .tile_traversal = {
        { .dimension = 0, .stride = A1L0Cfg::N, .wrap = A1L0Cfg::OUT_FEAT / A1L0Cfg::N },
        { .dimension = 1, .stride = A1L0Cfg::M, .wrap = A1L0Cfg::padded_independent_extent / A1L0Cfg::M }
      }
    });
    read_access(g1_buf_out.out[0]) = tiling({
      .buffer_dimension = { A1L0Cfg::OUT_FEAT, A1L0Cfg::padded_independent_extent },
      .tiling_dimension = { A1L0Cfg::OUT_FEAT, A1L0Cfg::padded_independent_extent },
      .offset = { 0, 0 },
      .boundary_dimension = { A1L0Cfg::OUT_FEAT, A1L0Cfg::padded_independent_extent }
    });
    connect<>(g1_buf_out.out[0], g1_ofm[0]);

    for (int ch = 0; ch < A1L0Cfg::CAS_NUM; ++ch) {
      for (int col = 0; col < A1L0Cfg::CAS_LENGTH; ++col)
        connect<>(g1_wts0[ch * A1L0Cfg::CAS_LENGTH + col], g1_l0.wts[ch * A1L0Cfg::CAS_LENGTH + col]);
      connect<>(g1_bias0[ch], g1_l0.bias[ch]);
    }

    g1_l0.place_graph(A1L0Cfg::col_placement, A1L0Cfg::row_placement);

    // ── Group 2 ──
    g2_buf_in = shared_buffer<typename A2L0Cfg::data_t>::create(
      { A2L0Cfg::IN_FEAT, A2L0Cfg::padded_independent_extent }, 1, 1);
    num_buffers(g2_buf_in) = 2;
    connect<>(g2_ifm[0], g2_buf_in.in[0]);
    write_access(g2_buf_in.in[0]) = tiling({
      .buffer_dimension = { A2L0Cfg::IN_FEAT, A2L0Cfg::padded_independent_extent },
      .tiling_dimension = { A2L0Cfg::IN_FEAT, A2L0Cfg::padded_independent_extent },
      .offset = { 0, 0 }
    });
    read_access(g2_buf_in.out[0]) = tiling({
      .buffer_dimension = { A2L0Cfg::IN_FEAT, A2L0Cfg::padded_independent_extent },
      .tiling_dimension = { A2L0Cfg::K, A2L0Cfg::M },
      .offset = { 0, 0 },
      .tile_traversal = {
        { .dimension = 0, .stride = A2L0Cfg::K, .wrap = A2L0Cfg::IN_FEAT / A2L0Cfg::K },
        { .dimension = 1, .stride = A2L0Cfg::M, .wrap = A2L0Cfg::padded_independent_extent / A2L0Cfg::M }
      },
      .boundary_dimension = { A2L0Cfg::IN_FEAT, A2L0Cfg::padded_independent_extent }
    });
    connect<>(g2_buf_in.out[0], g2_l0.in1[0]);

    g2_buf_out = shared_buffer<typename A2L0Cfg::result_t>::create(
      { A2L0Cfg::OUT_FEAT, A2L0Cfg::padded_independent_extent }, 1, 1);
    num_buffers(g2_buf_out) = 2;
    connect<>(g2_l0.out1[0], g2_buf_out.in[0]);
    write_access(g2_buf_out.in[0]) = tiling({
      .buffer_dimension = { A2L0Cfg::OUT_FEAT, A2L0Cfg::padded_independent_extent },
      .tiling_dimension = { A2L0Cfg::N, A2L0Cfg::M },
      .offset = { 0, 0 },
      .tile_traversal = {
        { .dimension = 0, .stride = A2L0Cfg::N, .wrap = A2L0Cfg::OUT_FEAT / A2L0Cfg::N },
        { .dimension = 1, .stride = A2L0Cfg::M, .wrap = A2L0Cfg::padded_independent_extent / A2L0Cfg::M }
      }
    });
    read_access(g2_buf_out.out[0]) = tiling({
      .buffer_dimension = { A2L0Cfg::OUT_FEAT, A2L0Cfg::padded_independent_extent },
      .tiling_dimension = { A2L0Cfg::OUT_FEAT, A2L0Cfg::padded_independent_extent },
      .offset = { 0, 0 },
      .boundary_dimension = { A2L0Cfg::OUT_FEAT, A2L0Cfg::padded_independent_extent }
    });
    connect<>(g2_buf_out.out[0], g2_ofm[0]);

    for (int ch = 0; ch < A2L0Cfg::CAS_NUM; ++ch) {
      for (int col = 0; col < A2L0Cfg::CAS_LENGTH; ++col)
        connect<>(g2_wts0[ch * A2L0Cfg::CAS_LENGTH + col], g2_l0.wts[ch * A2L0Cfg::CAS_LENGTH + col]);
      connect<>(g2_bias0[ch], g2_l0.bias[ch]);
    }

    g2_l0.place_graph(A2L0Cfg::col_placement, A2L0Cfg::row_placement);

    // ── Group 3 ──
    g3_buf_in = shared_buffer<typename A3L0Cfg::data_t>::create(
      { A3L0Cfg::IN_FEAT, A3L0Cfg::padded_independent_extent }, 1, 1);
    num_buffers(g3_buf_in) = 2;
    connect<>(g3_ifm[0], g3_buf_in.in[0]);
    write_access(g3_buf_in.in[0]) = tiling({
      .buffer_dimension = { A3L0Cfg::IN_FEAT, A3L0Cfg::padded_independent_extent },
      .tiling_dimension = { A3L0Cfg::IN_FEAT, A3L0Cfg::padded_independent_extent },
      .offset = { 0, 0 }
    });
    read_access(g3_buf_in.out[0]) = tiling({
      .buffer_dimension = { A3L0Cfg::IN_FEAT, A3L0Cfg::padded_independent_extent },
      .tiling_dimension = { A3L0Cfg::K, A3L0Cfg::M },
      .offset = { 0, 0 },
      .tile_traversal = {
        { .dimension = 0, .stride = A3L0Cfg::K, .wrap = A3L0Cfg::IN_FEAT / A3L0Cfg::K },
        { .dimension = 1, .stride = A3L0Cfg::M, .wrap = A3L0Cfg::padded_independent_extent / A3L0Cfg::M }
      },
      .boundary_dimension = { A3L0Cfg::IN_FEAT, A3L0Cfg::padded_independent_extent }
    });
    connect<>(g3_buf_in.out[0], g3_l0.in1[0]);

    g3_buf_out = shared_buffer<typename A3L0Cfg::result_t>::create(
      { A3L0Cfg::OUT_FEAT, A3L0Cfg::padded_independent_extent }, 1, 1);
    num_buffers(g3_buf_out) = 2;
    connect<>(g3_l0.out1[0], g3_buf_out.in[0]);
    write_access(g3_buf_out.in[0]) = tiling({
      .buffer_dimension = { A3L0Cfg::OUT_FEAT, A3L0Cfg::padded_independent_extent },
      .tiling_dimension = { A3L0Cfg::N, A3L0Cfg::M },
      .offset = { 0, 0 },
      .tile_traversal = {
        { .dimension = 0, .stride = A3L0Cfg::N, .wrap = A3L0Cfg::OUT_FEAT / A3L0Cfg::N },
        { .dimension = 1, .stride = A3L0Cfg::M, .wrap = A3L0Cfg::padded_independent_extent / A3L0Cfg::M }
      }
    });
    read_access(g3_buf_out.out[0]) = tiling({
      .buffer_dimension = { A3L0Cfg::OUT_FEAT, A3L0Cfg::padded_independent_extent },
      .tiling_dimension = { A3L0Cfg::OUT_FEAT, A3L0Cfg::padded_independent_extent },
      .offset = { 0, 0 },
      .boundary_dimension = { A3L0Cfg::OUT_FEAT, A3L0Cfg::padded_independent_extent }
    });
    connect<>(g3_buf_out.out[0], g3_ofm[0]);

    for (int ch = 0; ch < A3L0Cfg::CAS_NUM; ++ch) {
      for (int col = 0; col < A3L0Cfg::CAS_LENGTH; ++col)
        connect<>(g3_wts0[ch * A3L0Cfg::CAS_LENGTH + col], g3_l0.wts[ch * A3L0Cfg::CAS_LENGTH + col]);
      connect<>(g3_bias0[ch], g3_l0.bias[ch]);
    }

    g3_l0.place_graph(A3L0Cfg::col_placement, A3L0Cfg::row_placement);

    // ── Group 4 ──
    g4_buf_in = shared_buffer<typename A4L0Cfg::data_t>::create(
      { A4L0Cfg::IN_FEAT, A4L0Cfg::padded_independent_extent }, 1, 1);
    num_buffers(g4_buf_in) = 2;
    connect<>(g4_ifm[0], g4_buf_in.in[0]);
    write_access(g4_buf_in.in[0]) = tiling({
      .buffer_dimension = { A4L0Cfg::IN_FEAT, A4L0Cfg::padded_independent_extent },
      .tiling_dimension = { A4L0Cfg::IN_FEAT, A4L0Cfg::padded_independent_extent },
      .offset = { 0, 0 }
    });
    read_access(g4_buf_in.out[0]) = tiling({
      .buffer_dimension = { A4L0Cfg::IN_FEAT, A4L0Cfg::padded_independent_extent },
      .tiling_dimension = { A4L0Cfg::K, A4L0Cfg::M },
      .offset = { 0, 0 },
      .tile_traversal = {
        { .dimension = 0, .stride = A4L0Cfg::K, .wrap = A4L0Cfg::IN_FEAT / A4L0Cfg::K },
        { .dimension = 1, .stride = A4L0Cfg::M, .wrap = A4L0Cfg::padded_independent_extent / A4L0Cfg::M }
      },
      .boundary_dimension = { A4L0Cfg::IN_FEAT, A4L0Cfg::padded_independent_extent }
    });
    connect<>(g4_buf_in.out[0], g4_l0.in1[0]);

    g4_buf_out = shared_buffer<typename A4L0Cfg::result_t>::create(
      { A4L0Cfg::OUT_FEAT, A4L0Cfg::padded_independent_extent }, 1, 1);
    num_buffers(g4_buf_out) = 2;
    connect<>(g4_l0.out1[0], g4_buf_out.in[0]);
    write_access(g4_buf_out.in[0]) = tiling({
      .buffer_dimension = { A4L0Cfg::OUT_FEAT, A4L0Cfg::padded_independent_extent },
      .tiling_dimension = { A4L0Cfg::N, A4L0Cfg::M },
      .offset = { 0, 0 },
      .tile_traversal = {
        { .dimension = 0, .stride = A4L0Cfg::N, .wrap = A4L0Cfg::OUT_FEAT / A4L0Cfg::N },
        { .dimension = 1, .stride = A4L0Cfg::M, .wrap = A4L0Cfg::padded_independent_extent / A4L0Cfg::M }
      }
    });
    read_access(g4_buf_out.out[0]) = tiling({
      .buffer_dimension = { A4L0Cfg::OUT_FEAT, A4L0Cfg::padded_independent_extent },
      .tiling_dimension = { A4L0Cfg::OUT_FEAT, A4L0Cfg::padded_independent_extent },
      .offset = { 0, 0 },
      .boundary_dimension = { A4L0Cfg::OUT_FEAT, A4L0Cfg::padded_independent_extent }
    });
    connect<>(g4_buf_out.out[0], g4_ofm[0]);

    for (int ch = 0; ch < A4L0Cfg::CAS_NUM; ++ch) {
      for (int col = 0; col < A4L0Cfg::CAS_LENGTH; ++col)
        connect<>(g4_wts0[ch * A4L0Cfg::CAS_LENGTH + col], g4_l0.wts[ch * A4L0Cfg::CAS_LENGTH + col]);
      connect<>(g4_bias0[ch], g4_l0.bias[ch]);
    }

    g4_l0.place_graph(A4L0Cfg::col_placement, A4L0Cfg::row_placement);

    // ── Group 5 ──
    g5_buf_in = shared_buffer<typename A5L0Cfg::data_t>::create(
      { A5L0Cfg::IN_FEAT, A5L0Cfg::padded_independent_extent }, 1, 1);
    num_buffers(g5_buf_in) = 2;
    connect<>(g5_ifm[0], g5_buf_in.in[0]);
    write_access(g5_buf_in.in[0]) = tiling({
      .buffer_dimension = { A5L0Cfg::IN_FEAT, A5L0Cfg::padded_independent_extent },
      .tiling_dimension = { A5L0Cfg::IN_FEAT, A5L0Cfg::padded_independent_extent },
      .offset = { 0, 0 }
    });
    read_access(g5_buf_in.out[0]) = tiling({
      .buffer_dimension = { A5L0Cfg::IN_FEAT, A5L0Cfg::padded_independent_extent },
      .tiling_dimension = { A5L0Cfg::K, A5L0Cfg::M },
      .offset = { 0, 0 },
      .tile_traversal = {
        { .dimension = 0, .stride = A5L0Cfg::K, .wrap = A5L0Cfg::IN_FEAT / A5L0Cfg::K },
        { .dimension = 1, .stride = A5L0Cfg::M, .wrap = A5L0Cfg::padded_independent_extent / A5L0Cfg::M }
      },
      .boundary_dimension = { A5L0Cfg::IN_FEAT, A5L0Cfg::padded_independent_extent }
    });
    connect<>(g5_buf_in.out[0], g5_l0.in1[0]);

    g5_buf_out = shared_buffer<typename A5L0Cfg::result_t>::create(
      { A5L0Cfg::OUT_FEAT, A5L0Cfg::padded_independent_extent }, 1, 1);
    num_buffers(g5_buf_out) = 2;
    connect<>(g5_l0.out1[0], g5_buf_out.in[0]);
    write_access(g5_buf_out.in[0]) = tiling({
      .buffer_dimension = { A5L0Cfg::OUT_FEAT, A5L0Cfg::padded_independent_extent },
      .tiling_dimension = { A5L0Cfg::N, A5L0Cfg::M },
      .offset = { 0, 0 },
      .tile_traversal = {
        { .dimension = 0, .stride = A5L0Cfg::N, .wrap = A5L0Cfg::OUT_FEAT / A5L0Cfg::N },
        { .dimension = 1, .stride = A5L0Cfg::M, .wrap = A5L0Cfg::padded_independent_extent / A5L0Cfg::M }
      }
    });
    read_access(g5_buf_out.out[0]) = tiling({
      .buffer_dimension = { A5L0Cfg::OUT_FEAT, A5L0Cfg::padded_independent_extent },
      .tiling_dimension = { A5L0Cfg::OUT_FEAT, A5L0Cfg::padded_independent_extent },
      .offset = { 0, 0 },
      .boundary_dimension = { A5L0Cfg::OUT_FEAT, A5L0Cfg::padded_independent_extent }
    });
    connect<>(g5_buf_out.out[0], g5_ofm[0]);

    for (int ch = 0; ch < A5L0Cfg::CAS_NUM; ++ch) {
      for (int col = 0; col < A5L0Cfg::CAS_LENGTH; ++col)
        connect<>(g5_wts0[ch * A5L0Cfg::CAS_LENGTH + col], g5_l0.wts[ch * A5L0Cfg::CAS_LENGTH + col]);
      connect<>(g5_bias0[ch], g5_l0.bias[ch]);
    }

    g5_l0.place_graph(A5L0Cfg::col_placement, A5L0Cfg::row_placement);

    // ── Group 6 ──
    g6_buf_in = shared_buffer<typename A6L0Cfg::data_t>::create(
      { A6L0Cfg::IN_FEAT, A6L0Cfg::padded_independent_extent }, 1, 1);
    num_buffers(g6_buf_in) = 2;
    connect<>(g6_ifm[0], g6_buf_in.in[0]);
    write_access(g6_buf_in.in[0]) = tiling({
      .buffer_dimension = { A6L0Cfg::IN_FEAT, A6L0Cfg::padded_independent_extent },
      .tiling_dimension = { A6L0Cfg::IN_FEAT, A6L0Cfg::padded_independent_extent },
      .offset = { 0, 0 }
    });
    read_access(g6_buf_in.out[0]) = tiling({
      .buffer_dimension = { A6L0Cfg::IN_FEAT, A6L0Cfg::padded_independent_extent },
      .tiling_dimension = { A6L0Cfg::K, A6L0Cfg::M },
      .offset = { 0, 0 },
      .tile_traversal = {
        { .dimension = 0, .stride = A6L0Cfg::K, .wrap = A6L0Cfg::IN_FEAT / A6L0Cfg::K },
        { .dimension = 1, .stride = A6L0Cfg::M, .wrap = A6L0Cfg::padded_independent_extent / A6L0Cfg::M }
      },
      .boundary_dimension = { A6L0Cfg::IN_FEAT, A6L0Cfg::padded_independent_extent }
    });
    connect<>(g6_buf_in.out[0], g6_l0.in1[0]);

    g6_buf_out = shared_buffer<typename A6L0Cfg::result_t>::create(
      { A6L0Cfg::OUT_FEAT, A6L0Cfg::padded_independent_extent }, 1, 1);
    num_buffers(g6_buf_out) = 2;
    connect<>(g6_l0.out1[0], g6_buf_out.in[0]);
    write_access(g6_buf_out.in[0]) = tiling({
      .buffer_dimension = { A6L0Cfg::OUT_FEAT, A6L0Cfg::padded_independent_extent },
      .tiling_dimension = { A6L0Cfg::N, A6L0Cfg::M },
      .offset = { 0, 0 },
      .tile_traversal = {
        { .dimension = 0, .stride = A6L0Cfg::N, .wrap = A6L0Cfg::OUT_FEAT / A6L0Cfg::N },
        { .dimension = 1, .stride = A6L0Cfg::M, .wrap = A6L0Cfg::padded_independent_extent / A6L0Cfg::M }
      }
    });
    read_access(g6_buf_out.out[0]) = tiling({
      .buffer_dimension = { A6L0Cfg::OUT_FEAT, A6L0Cfg::padded_independent_extent },
      .tiling_dimension = { A6L0Cfg::OUT_FEAT, A6L0Cfg::padded_independent_extent },
      .offset = { 0, 0 },
      .boundary_dimension = { A6L0Cfg::OUT_FEAT, A6L0Cfg::padded_independent_extent }
    });
    connect<>(g6_buf_out.out[0], g6_ofm[0]);

    for (int ch = 0; ch < A6L0Cfg::CAS_NUM; ++ch) {
      for (int col = 0; col < A6L0Cfg::CAS_LENGTH; ++col)
        connect<>(g6_wts0[ch * A6L0Cfg::CAS_LENGTH + col], g6_l0.wts[ch * A6L0Cfg::CAS_LENGTH + col]);
      connect<>(g6_bias0[ch], g6_l0.bias[ch]);
    }

    g6_l0.place_graph(A6L0Cfg::col_placement, A6L0Cfg::row_placement);

  }
};

class dut_graph : public graph {
public:
  input_plio  plio_g0_in;
  output_plio plio_g0_out;
  input_plio  plio_g1_in;
  output_plio plio_g1_out;
  input_plio  plio_g2_in;
  output_plio plio_g2_out;
  input_plio  plio_g3_in;
  output_plio plio_g3_out;
  input_plio  plio_g4_in;
  output_plio plio_g4_out;
  input_plio  plio_g5_in;
  output_plio plio_g5_out;
  input_plio  plio_g6_in;
  output_plio plio_g6_out;

  input_port g0_wts0[A0L0Cfg::CAS_NUM * A0L0Cfg::CAS_LENGTH];
  input_port g0_bias0[A0L0Cfg::CAS_NUM];
  input_port g0_wts1[A0L1Cfg::CAS_NUM * A0L1Cfg::CAS_LENGTH];
  input_port g0_bias1[A0L1Cfg::CAS_NUM];
  input_port g1_wts0[A1L0Cfg::CAS_NUM * A1L0Cfg::CAS_LENGTH];
  input_port g1_bias0[A1L0Cfg::CAS_NUM];
  input_port g2_wts0[A2L0Cfg::CAS_NUM * A2L0Cfg::CAS_LENGTH];
  input_port g2_bias0[A2L0Cfg::CAS_NUM];
  input_port g3_wts0[A3L0Cfg::CAS_NUM * A3L0Cfg::CAS_LENGTH];
  input_port g3_bias0[A3L0Cfg::CAS_NUM];
  input_port g4_wts0[A4L0Cfg::CAS_NUM * A4L0Cfg::CAS_LENGTH];
  input_port g4_bias0[A4L0Cfg::CAS_NUM];
  input_port g5_wts0[A5L0Cfg::CAS_NUM * A5L0Cfg::CAS_LENGTH];
  input_port g5_bias0[A5L0Cfg::CAS_NUM];
  input_port g6_wts0[A6L0Cfg::CAS_NUM * A6L0Cfg::CAS_LENGTH];
  input_port g6_bias0[A6L0Cfg::CAS_NUM];

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
    for (int ch = 0; ch < A0L1Cfg::CAS_NUM; ++ch) {
      for (int col = 0; col < A0L1Cfg::CAS_LENGTH; ++col)
        connect<>(g0_wts1[ch * A0L1Cfg::CAS_LENGTH + col], dut.g0_wts1[ch * A0L1Cfg::CAS_LENGTH + col]);
      connect<>(g0_bias1[ch], dut.g0_bias1[ch]);
    }

    plio_g1_in  = input_plio::create("g1_in",  plio_128_bits, "data/g1_ifm.txt");
    plio_g1_out = output_plio::create("g1_out", plio_128_bits, "data/g1_ofm.txt");
    connect<>(plio_g1_in.out[0],  dut.g1_ifm[0]);
    connect<>(dut.g1_ofm[0], plio_g1_out.in[0]);

    for (int ch = 0; ch < A1L0Cfg::CAS_NUM; ++ch) {
      for (int col = 0; col < A1L0Cfg::CAS_LENGTH; ++col)
        connect<>(g1_wts0[ch * A1L0Cfg::CAS_LENGTH + col], dut.g1_wts0[ch * A1L0Cfg::CAS_LENGTH + col]);
      connect<>(g1_bias0[ch], dut.g1_bias0[ch]);
    }

    plio_g2_in  = input_plio::create("g2_in",  plio_128_bits, "data/g2_ifm.txt");
    plio_g2_out = output_plio::create("g2_out", plio_128_bits, "data/g2_ofm.txt");
    connect<>(plio_g2_in.out[0],  dut.g2_ifm[0]);
    connect<>(dut.g2_ofm[0], plio_g2_out.in[0]);

    for (int ch = 0; ch < A2L0Cfg::CAS_NUM; ++ch) {
      for (int col = 0; col < A2L0Cfg::CAS_LENGTH; ++col)
        connect<>(g2_wts0[ch * A2L0Cfg::CAS_LENGTH + col], dut.g2_wts0[ch * A2L0Cfg::CAS_LENGTH + col]);
      connect<>(g2_bias0[ch], dut.g2_bias0[ch]);
    }

    plio_g3_in  = input_plio::create("g3_in",  plio_128_bits, "data/g3_ifm.txt");
    plio_g3_out = output_plio::create("g3_out", plio_128_bits, "data/g3_ofm.txt");
    connect<>(plio_g3_in.out[0],  dut.g3_ifm[0]);
    connect<>(dut.g3_ofm[0], plio_g3_out.in[0]);

    for (int ch = 0; ch < A3L0Cfg::CAS_NUM; ++ch) {
      for (int col = 0; col < A3L0Cfg::CAS_LENGTH; ++col)
        connect<>(g3_wts0[ch * A3L0Cfg::CAS_LENGTH + col], dut.g3_wts0[ch * A3L0Cfg::CAS_LENGTH + col]);
      connect<>(g3_bias0[ch], dut.g3_bias0[ch]);
    }

    plio_g4_in  = input_plio::create("g4_in",  plio_128_bits, "data/g4_ifm.txt");
    plio_g4_out = output_plio::create("g4_out", plio_128_bits, "data/g4_ofm.txt");
    connect<>(plio_g4_in.out[0],  dut.g4_ifm[0]);
    connect<>(dut.g4_ofm[0], plio_g4_out.in[0]);

    for (int ch = 0; ch < A4L0Cfg::CAS_NUM; ++ch) {
      for (int col = 0; col < A4L0Cfg::CAS_LENGTH; ++col)
        connect<>(g4_wts0[ch * A4L0Cfg::CAS_LENGTH + col], dut.g4_wts0[ch * A4L0Cfg::CAS_LENGTH + col]);
      connect<>(g4_bias0[ch], dut.g4_bias0[ch]);
    }

    plio_g5_in  = input_plio::create("g5_in",  plio_128_bits, "data/g5_ifm.txt");
    plio_g5_out = output_plio::create("g5_out", plio_128_bits, "data/g5_ofm.txt");
    connect<>(plio_g5_in.out[0],  dut.g5_ifm[0]);
    connect<>(dut.g5_ofm[0], plio_g5_out.in[0]);

    for (int ch = 0; ch < A5L0Cfg::CAS_NUM; ++ch) {
      for (int col = 0; col < A5L0Cfg::CAS_LENGTH; ++col)
        connect<>(g5_wts0[ch * A5L0Cfg::CAS_LENGTH + col], dut.g5_wts0[ch * A5L0Cfg::CAS_LENGTH + col]);
      connect<>(g5_bias0[ch], dut.g5_bias0[ch]);
    }

    plio_g6_in  = input_plio::create("g6_in",  plio_128_bits, "data/g6_ifm.txt");
    plio_g6_out = output_plio::create("g6_out", plio_128_bits, "data/g6_ofm.txt");
    connect<>(plio_g6_in.out[0],  dut.g6_ifm[0]);
    connect<>(dut.g6_ofm[0], plio_g6_out.in[0]);

    for (int ch = 0; ch < A6L0Cfg::CAS_NUM; ++ch) {
      for (int col = 0; col < A6L0Cfg::CAS_LENGTH; ++col)
        connect<>(g6_wts0[ch * A6L0Cfg::CAS_LENGTH + col], dut.g6_wts0[ch * A6L0Cfg::CAS_LENGTH + col]);
      connect<>(g6_bias0[ch], dut.g6_bias0[ch]);
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
  for (int ch = 0; ch < A0L1Cfg::CAS_NUM; ++ch) {
    for (int col = 0; col < A0L1Cfg::CAS_LENGTH; ++col) {
      int idx = ch * A0L1Cfg::CAS_LENGTH + col;
      dut.update(dut.g0_wts1[idx], g0_w1[ch][col], A0L1Cfg::IN_FEAT_SLICE * A0L1Cfg::OUT_FEAT_SLICE);
    }
    dut.update(dut.g0_bias1[ch], g0_b1[ch], A0L1Cfg::OUT_FEAT_SLICE);
  }
  for (int ch = 0; ch < A1L0Cfg::CAS_NUM; ++ch) {
    for (int col = 0; col < A1L0Cfg::CAS_LENGTH; ++col) {
      int idx = ch * A1L0Cfg::CAS_LENGTH + col;
      dut.update(dut.g1_wts0[idx], g1_w0[ch][col], A1L0Cfg::IN_FEAT_SLICE * A1L0Cfg::OUT_FEAT_SLICE);
    }
    dut.update(dut.g1_bias0[ch], g1_b0[ch], A1L0Cfg::OUT_FEAT_SLICE);
  }
  for (int ch = 0; ch < A2L0Cfg::CAS_NUM; ++ch) {
    for (int col = 0; col < A2L0Cfg::CAS_LENGTH; ++col) {
      int idx = ch * A2L0Cfg::CAS_LENGTH + col;
      dut.update(dut.g2_wts0[idx], g2_w0[ch][col], A2L0Cfg::IN_FEAT_SLICE * A2L0Cfg::OUT_FEAT_SLICE);
    }
    dut.update(dut.g2_bias0[ch], g2_b0[ch], A2L0Cfg::OUT_FEAT_SLICE);
  }
  for (int ch = 0; ch < A3L0Cfg::CAS_NUM; ++ch) {
    for (int col = 0; col < A3L0Cfg::CAS_LENGTH; ++col) {
      int idx = ch * A3L0Cfg::CAS_LENGTH + col;
      dut.update(dut.g3_wts0[idx], g3_w0[ch][col], A3L0Cfg::IN_FEAT_SLICE * A3L0Cfg::OUT_FEAT_SLICE);
    }
    dut.update(dut.g3_bias0[ch], g3_b0[ch], A3L0Cfg::OUT_FEAT_SLICE);
  }
  for (int ch = 0; ch < A4L0Cfg::CAS_NUM; ++ch) {
    for (int col = 0; col < A4L0Cfg::CAS_LENGTH; ++col) {
      int idx = ch * A4L0Cfg::CAS_LENGTH + col;
      dut.update(dut.g4_wts0[idx], g4_w0[ch][col], A4L0Cfg::IN_FEAT_SLICE * A4L0Cfg::OUT_FEAT_SLICE);
    }
    dut.update(dut.g4_bias0[ch], g4_b0[ch], A4L0Cfg::OUT_FEAT_SLICE);
  }
  for (int ch = 0; ch < A5L0Cfg::CAS_NUM; ++ch) {
    for (int col = 0; col < A5L0Cfg::CAS_LENGTH; ++col) {
      int idx = ch * A5L0Cfg::CAS_LENGTH + col;
      dut.update(dut.g5_wts0[idx], g5_w0[ch][col], A5L0Cfg::IN_FEAT_SLICE * A5L0Cfg::OUT_FEAT_SLICE);
    }
    dut.update(dut.g5_bias0[ch], g5_b0[ch], A5L0Cfg::OUT_FEAT_SLICE);
  }
  for (int ch = 0; ch < A6L0Cfg::CAS_NUM; ++ch) {
    for (int col = 0; col < A6L0Cfg::CAS_LENGTH; ++col) {
      int idx = ch * A6L0Cfg::CAS_LENGTH + col;
      dut.update(dut.g6_wts0[idx], g6_w0[ch][col], A6L0Cfg::IN_FEAT_SLICE * A6L0Cfg::OUT_FEAT_SLICE);
    }
    dut.update(dut.g6_bias0[ch], g6_b0[ch], A6L0Cfg::OUT_FEAT_SLICE);
  }

#ifdef __AIESIM__
  event::handle h0 = event::start_profiling(
    dut.plio_g0_in, dut.plio_g0_out,
    event::io_stream_start_difference_cycles);
  event::handle h1 = event::start_profiling(
    dut.plio_g1_in, dut.plio_g1_out,
    event::io_stream_start_difference_cycles);
  event::handle h2 = event::start_profiling(
    dut.plio_g2_in, dut.plio_g2_out,
    event::io_stream_start_difference_cycles);
  event::handle h3 = event::start_profiling(
    dut.plio_g3_in, dut.plio_g3_out,
    event::io_stream_start_difference_cycles);
  event::handle h4 = event::start_profiling(
    dut.plio_g4_in, dut.plio_g4_out,
    event::io_stream_start_difference_cycles);
  event::handle h5 = event::start_profiling(
    dut.plio_g5_in, dut.plio_g5_out,
    event::io_stream_start_difference_cycles);
  event::handle h6 = event::start_profiling(
    dut.plio_g6_in, dut.plio_g6_out,
    event::io_stream_start_difference_cycles);
#endif

  dut.run(N_ITER);
  dut.wait();

#ifdef __AIESIM__
  long long cyc0 = event::read_profiling(h0);
  event::stop_profiling(h0);
  long long cyc1 = event::read_profiling(h1);
  event::stop_profiling(h1);
  long long cyc2 = event::read_profiling(h2);
  event::stop_profiling(h2);
  long long cyc3 = event::read_profiling(h3);
  event::stop_profiling(h3);
  long long cyc4 = event::read_profiling(h4);
  event::stop_profiling(h4);
  long long cyc5 = event::read_profiling(h5);
  event::stop_profiling(h5);
  long long cyc6 = event::read_profiling(h6);
  event::stop_profiling(h6);
  std::system("mkdir -p aiesimulator_output/data");
  { std::ofstream lf("aiesimulator_output/data/g0_latency.json");
    lf << "{\"cycles\": " << cyc0 << "}\n"; }
  { std::ofstream lf("aiesimulator_output/data/g1_latency.json");
    lf << "{\"cycles\": " << cyc1 << "}\n"; }
  { std::ofstream lf("aiesimulator_output/data/g2_latency.json");
    lf << "{\"cycles\": " << cyc2 << "}\n"; }
  { std::ofstream lf("aiesimulator_output/data/g3_latency.json");
    lf << "{\"cycles\": " << cyc3 << "}\n"; }
  { std::ofstream lf("aiesimulator_output/data/g4_latency.json");
    lf << "{\"cycles\": " << cyc4 << "}\n"; }
  { std::ofstream lf("aiesimulator_output/data/g5_latency.json");
    lf << "{\"cycles\": " << cyc5 << "}\n"; }
  { std::ofstream lf("aiesimulator_output/data/g6_latency.json");
    lf << "{\"cycles\": " << cyc6 << "}\n"; }
#endif

  dut.end();
  return 0;
}
#endif
