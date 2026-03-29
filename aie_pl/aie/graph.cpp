#include <adf.h>
#include <fstream>
#include "parameters.h"
#include "dense_graph.h"

extern "C" {
  #include "weights/g0_w0.h"
  #include "weights/g0_b0.h"
  #include "weights/g0_w1.h"
  #include "weights/g0_b1.h"
  #include "weights/g0_w2.h"
  #include "weights/g0_b2.h"
  #include "weights/g0_w3.h"
  #include "weights/g0_b3.h"
  #include "weights/g0_w4.h"
  #include "weights/g0_b4.h"
  #include "weights/g0_w5.h"
  #include "weights/g0_b5.h"
  #include "weights/g0_w6.h"
  #include "weights/g0_b6.h"
  #include "weights/g0_w7.h"
  #include "weights/g0_b7.h"
}

using namespace adf;

class top_graph : public graph {
public:
  input_port  g0_ifm[1];
  output_port g0_ofm[1];

  input_port g0_wts0[A0L0Cfg::CAS_NUM * A0L0Cfg::CAS_LENGTH];
  input_port g0_bias0[A0L0Cfg::CAS_NUM];
  input_port g0_wts1[A0L1Cfg::CAS_NUM * A0L1Cfg::CAS_LENGTH];
  input_port g0_bias1[A0L1Cfg::CAS_NUM];
  input_port g0_wts2[A0L2Cfg::CAS_NUM * A0L2Cfg::CAS_LENGTH];
  input_port g0_bias2[A0L2Cfg::CAS_NUM];
  input_port g0_wts3[A0L3Cfg::CAS_NUM * A0L3Cfg::CAS_LENGTH];
  input_port g0_bias3[A0L3Cfg::CAS_NUM];
  input_port g0_wts4[A0L4Cfg::CAS_NUM * A0L4Cfg::CAS_LENGTH];
  input_port g0_bias4[A0L4Cfg::CAS_NUM];
  input_port g0_wts5[A0L5Cfg::CAS_NUM * A0L5Cfg::CAS_LENGTH];
  input_port g0_bias5[A0L5Cfg::CAS_NUM];
  input_port g0_wts6[A0L6Cfg::CAS_NUM * A0L6Cfg::CAS_LENGTH];
  input_port g0_bias6[A0L6Cfg::CAS_NUM];
  input_port g0_wts7[A0L7Cfg::CAS_NUM * A0L7Cfg::CAS_LENGTH];
  input_port g0_bias7[A0L7Cfg::CAS_NUM];

private:
  dense_bias_relu_graph<A0L0Cfg> g0_l0;
  dense_bias_relu_graph<A0L1Cfg> g0_l1;
  dense_bias_relu_graph<A0L2Cfg> g0_l2;
  dense_bias_relu_graph<A0L3Cfg> g0_l3;
  dense_bias_relu_graph<A0L4Cfg> g0_l4;
  dense_bias_relu_graph<A0L5Cfg> g0_l5;
  dense_bias_relu_graph<A0L6Cfg> g0_l6;
  dense_bias_relu_graph<A0L7Cfg> g0_l7;

  shared_buffer<typename A0L0Cfg::data_t>    g0_buf_in;
  shared_buffer<typename A0L0Cfg::result_t>  g0_buf_mid0;
  shared_buffer<typename A0L1Cfg::result_t>  g0_buf_mid1;
  shared_buffer<typename A0L2Cfg::result_t>  g0_buf_mid2;
  shared_buffer<typename A0L3Cfg::result_t>  g0_buf_mid3;
  shared_buffer<typename A0L4Cfg::result_t>  g0_buf_mid4;
  shared_buffer<typename A0L5Cfg::result_t>  g0_buf_mid5;
  shared_buffer<typename A0L6Cfg::result_t>  g0_buf_mid6;
  shared_buffer<typename A0L7Cfg::result_t> g0_buf_out;

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

    g0_buf_mid1 = shared_buffer<typename A0L1Cfg::result_t>::create(
      { A0L1Cfg::OUT_FEAT, A0L1Cfg::padded_independent_extent }, 1, 1);
    num_buffers(g0_buf_mid1) = 2;
    connect<>(g0_l1.out1[0], g0_buf_mid1.in[0]);
    write_access(g0_buf_mid1.in[0]) = tiling({
      .buffer_dimension = { A0L1Cfg::OUT_FEAT, A0L1Cfg::padded_independent_extent },
      .tiling_dimension = { A0L1Cfg::N, A0L1Cfg::M },
      .offset = { 0, 0 },
      .tile_traversal = {
        { .dimension = 0, .stride = A0L1Cfg::N, .wrap = A0L1Cfg::OUT_FEAT / A0L1Cfg::N },
        { .dimension = 1, .stride = A0L1Cfg::M, .wrap = A0L1Cfg::padded_independent_extent / A0L1Cfg::M }
      }
    });
    read_access(g0_buf_mid1.out[0]) = tiling({
      .buffer_dimension = { A0L2Cfg::IN_FEAT, A0L2Cfg::padded_independent_extent },
      .tiling_dimension = { A0L2Cfg::K, A0L2Cfg::M },
      .offset = { 0, 0 },
      .tile_traversal = {
        { .dimension = 0, .stride = A0L2Cfg::K, .wrap = A0L2Cfg::IN_FEAT / A0L2Cfg::K },
        { .dimension = 1, .stride = A0L2Cfg::M, .wrap = A0L2Cfg::padded_independent_extent / A0L2Cfg::M }
      },
      .boundary_dimension = { A0L2Cfg::IN_FEAT, A0L2Cfg::padded_independent_extent }
    });
    connect<>(g0_buf_mid1.out[0], g0_l2.in1[0]);

    g0_buf_mid2 = shared_buffer<typename A0L2Cfg::result_t>::create(
      { A0L2Cfg::OUT_FEAT, A0L2Cfg::padded_independent_extent }, 1, 1);
    num_buffers(g0_buf_mid2) = 2;
    connect<>(g0_l2.out1[0], g0_buf_mid2.in[0]);
    write_access(g0_buf_mid2.in[0]) = tiling({
      .buffer_dimension = { A0L2Cfg::OUT_FEAT, A0L2Cfg::padded_independent_extent },
      .tiling_dimension = { A0L2Cfg::N, A0L2Cfg::M },
      .offset = { 0, 0 },
      .tile_traversal = {
        { .dimension = 0, .stride = A0L2Cfg::N, .wrap = A0L2Cfg::OUT_FEAT / A0L2Cfg::N },
        { .dimension = 1, .stride = A0L2Cfg::M, .wrap = A0L2Cfg::padded_independent_extent / A0L2Cfg::M }
      }
    });
    read_access(g0_buf_mid2.out[0]) = tiling({
      .buffer_dimension = { A0L3Cfg::IN_FEAT, A0L3Cfg::padded_independent_extent },
      .tiling_dimension = { A0L3Cfg::K, A0L3Cfg::M },
      .offset = { 0, 0 },
      .tile_traversal = {
        { .dimension = 0, .stride = A0L3Cfg::K, .wrap = A0L3Cfg::IN_FEAT / A0L3Cfg::K },
        { .dimension = 1, .stride = A0L3Cfg::M, .wrap = A0L3Cfg::padded_independent_extent / A0L3Cfg::M }
      },
      .boundary_dimension = { A0L3Cfg::IN_FEAT, A0L3Cfg::padded_independent_extent }
    });
    connect<>(g0_buf_mid2.out[0], g0_l3.in1[0]);

    g0_buf_mid3 = shared_buffer<typename A0L3Cfg::result_t>::create(
      { A0L3Cfg::OUT_FEAT, A0L3Cfg::padded_independent_extent }, 1, 1);
    num_buffers(g0_buf_mid3) = 2;
    connect<>(g0_l3.out1[0], g0_buf_mid3.in[0]);
    write_access(g0_buf_mid3.in[0]) = tiling({
      .buffer_dimension = { A0L3Cfg::OUT_FEAT, A0L3Cfg::padded_independent_extent },
      .tiling_dimension = { A0L3Cfg::N, A0L3Cfg::M },
      .offset = { 0, 0 },
      .tile_traversal = {
        { .dimension = 0, .stride = A0L3Cfg::N, .wrap = A0L3Cfg::OUT_FEAT / A0L3Cfg::N },
        { .dimension = 1, .stride = A0L3Cfg::M, .wrap = A0L3Cfg::padded_independent_extent / A0L3Cfg::M }
      }
    });
    read_access(g0_buf_mid3.out[0]) = tiling({
      .buffer_dimension = { A0L4Cfg::IN_FEAT, A0L4Cfg::padded_independent_extent },
      .tiling_dimension = { A0L4Cfg::K, A0L4Cfg::M },
      .offset = { 0, 0 },
      .tile_traversal = {
        { .dimension = 0, .stride = A0L4Cfg::K, .wrap = A0L4Cfg::IN_FEAT / A0L4Cfg::K },
        { .dimension = 1, .stride = A0L4Cfg::M, .wrap = A0L4Cfg::padded_independent_extent / A0L4Cfg::M }
      },
      .boundary_dimension = { A0L4Cfg::IN_FEAT, A0L4Cfg::padded_independent_extent }
    });
    connect<>(g0_buf_mid3.out[0], g0_l4.in1[0]);

    g0_buf_mid4 = shared_buffer<typename A0L4Cfg::result_t>::create(
      { A0L4Cfg::OUT_FEAT, A0L4Cfg::padded_independent_extent }, 1, 1);
    num_buffers(g0_buf_mid4) = 2;
    connect<>(g0_l4.out1[0], g0_buf_mid4.in[0]);
    write_access(g0_buf_mid4.in[0]) = tiling({
      .buffer_dimension = { A0L4Cfg::OUT_FEAT, A0L4Cfg::padded_independent_extent },
      .tiling_dimension = { A0L4Cfg::N, A0L4Cfg::M },
      .offset = { 0, 0 },
      .tile_traversal = {
        { .dimension = 0, .stride = A0L4Cfg::N, .wrap = A0L4Cfg::OUT_FEAT / A0L4Cfg::N },
        { .dimension = 1, .stride = A0L4Cfg::M, .wrap = A0L4Cfg::padded_independent_extent / A0L4Cfg::M }
      }
    });
    read_access(g0_buf_mid4.out[0]) = tiling({
      .buffer_dimension = { A0L5Cfg::IN_FEAT, A0L5Cfg::padded_independent_extent },
      .tiling_dimension = { A0L5Cfg::K, A0L5Cfg::M },
      .offset = { 0, 0 },
      .tile_traversal = {
        { .dimension = 0, .stride = A0L5Cfg::K, .wrap = A0L5Cfg::IN_FEAT / A0L5Cfg::K },
        { .dimension = 1, .stride = A0L5Cfg::M, .wrap = A0L5Cfg::padded_independent_extent / A0L5Cfg::M }
      },
      .boundary_dimension = { A0L5Cfg::IN_FEAT, A0L5Cfg::padded_independent_extent }
    });
    connect<>(g0_buf_mid4.out[0], g0_l5.in1[0]);

    g0_buf_mid5 = shared_buffer<typename A0L5Cfg::result_t>::create(
      { A0L5Cfg::OUT_FEAT, A0L5Cfg::padded_independent_extent }, 1, 1);
    num_buffers(g0_buf_mid5) = 2;
    connect<>(g0_l5.out1[0], g0_buf_mid5.in[0]);
    write_access(g0_buf_mid5.in[0]) = tiling({
      .buffer_dimension = { A0L5Cfg::OUT_FEAT, A0L5Cfg::padded_independent_extent },
      .tiling_dimension = { A0L5Cfg::N, A0L5Cfg::M },
      .offset = { 0, 0 },
      .tile_traversal = {
        { .dimension = 0, .stride = A0L5Cfg::N, .wrap = A0L5Cfg::OUT_FEAT / A0L5Cfg::N },
        { .dimension = 1, .stride = A0L5Cfg::M, .wrap = A0L5Cfg::padded_independent_extent / A0L5Cfg::M }
      }
    });
    read_access(g0_buf_mid5.out[0]) = tiling({
      .buffer_dimension = { A0L6Cfg::IN_FEAT, A0L6Cfg::padded_independent_extent },
      .tiling_dimension = { A0L6Cfg::K, A0L6Cfg::M },
      .offset = { 0, 0 },
      .tile_traversal = {
        { .dimension = 0, .stride = A0L6Cfg::K, .wrap = A0L6Cfg::IN_FEAT / A0L6Cfg::K },
        { .dimension = 1, .stride = A0L6Cfg::M, .wrap = A0L6Cfg::padded_independent_extent / A0L6Cfg::M }
      },
      .boundary_dimension = { A0L6Cfg::IN_FEAT, A0L6Cfg::padded_independent_extent }
    });
    connect<>(g0_buf_mid5.out[0], g0_l6.in1[0]);

    g0_buf_mid6 = shared_buffer<typename A0L6Cfg::result_t>::create(
      { A0L6Cfg::OUT_FEAT, A0L6Cfg::padded_independent_extent }, 1, 1);
    num_buffers(g0_buf_mid6) = 2;
    connect<>(g0_l6.out1[0], g0_buf_mid6.in[0]);
    write_access(g0_buf_mid6.in[0]) = tiling({
      .buffer_dimension = { A0L6Cfg::OUT_FEAT, A0L6Cfg::padded_independent_extent },
      .tiling_dimension = { A0L6Cfg::N, A0L6Cfg::M },
      .offset = { 0, 0 },
      .tile_traversal = {
        { .dimension = 0, .stride = A0L6Cfg::N, .wrap = A0L6Cfg::OUT_FEAT / A0L6Cfg::N },
        { .dimension = 1, .stride = A0L6Cfg::M, .wrap = A0L6Cfg::padded_independent_extent / A0L6Cfg::M }
      }
    });
    read_access(g0_buf_mid6.out[0]) = tiling({
      .buffer_dimension = { A0L7Cfg::IN_FEAT, A0L7Cfg::padded_independent_extent },
      .tiling_dimension = { A0L7Cfg::K, A0L7Cfg::M },
      .offset = { 0, 0 },
      .tile_traversal = {
        { .dimension = 0, .stride = A0L7Cfg::K, .wrap = A0L7Cfg::IN_FEAT / A0L7Cfg::K },
        { .dimension = 1, .stride = A0L7Cfg::M, .wrap = A0L7Cfg::padded_independent_extent / A0L7Cfg::M }
      },
      .boundary_dimension = { A0L7Cfg::IN_FEAT, A0L7Cfg::padded_independent_extent }
    });
    connect<>(g0_buf_mid6.out[0], g0_l7.in1[0]);

    g0_buf_out = shared_buffer<typename A0L7Cfg::result_t>::create(
      { A0L7Cfg::OUT_FEAT, A0L7Cfg::padded_independent_extent }, 1, 1);
    num_buffers(g0_buf_out) = 2;
    connect<>(g0_l7.out1[0], g0_buf_out.in[0]);
    write_access(g0_buf_out.in[0]) = tiling({
      .buffer_dimension = { A0L7Cfg::OUT_FEAT, A0L7Cfg::padded_independent_extent },
      .tiling_dimension = { A0L7Cfg::N, A0L7Cfg::M },
      .offset = { 0, 0 },
      .tile_traversal = {
        { .dimension = 0, .stride = A0L7Cfg::N, .wrap = A0L7Cfg::OUT_FEAT / A0L7Cfg::N },
        { .dimension = 1, .stride = A0L7Cfg::M, .wrap = A0L7Cfg::padded_independent_extent / A0L7Cfg::M }
      }
    });
    read_access(g0_buf_out.out[0]) = tiling({
      .buffer_dimension = { A0L7Cfg::OUT_FEAT, A0L7Cfg::padded_independent_extent },
      .tiling_dimension = { A0L7Cfg::OUT_FEAT, A0L7Cfg::padded_independent_extent },
      .offset = { 0, 0 },
      .boundary_dimension = { A0L7Cfg::OUT_FEAT, A0L7Cfg::padded_independent_extent }
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
    for (int ch = 0; ch < A0L2Cfg::CAS_NUM; ++ch) {
      for (int col = 0; col < A0L2Cfg::CAS_LENGTH; ++col)
        connect<>(g0_wts2[ch * A0L2Cfg::CAS_LENGTH + col], g0_l2.wts[ch * A0L2Cfg::CAS_LENGTH + col]);
      connect<>(g0_bias2[ch], g0_l2.bias[ch]);
    }
    for (int ch = 0; ch < A0L3Cfg::CAS_NUM; ++ch) {
      for (int col = 0; col < A0L3Cfg::CAS_LENGTH; ++col)
        connect<>(g0_wts3[ch * A0L3Cfg::CAS_LENGTH + col], g0_l3.wts[ch * A0L3Cfg::CAS_LENGTH + col]);
      connect<>(g0_bias3[ch], g0_l3.bias[ch]);
    }
    for (int ch = 0; ch < A0L4Cfg::CAS_NUM; ++ch) {
      for (int col = 0; col < A0L4Cfg::CAS_LENGTH; ++col)
        connect<>(g0_wts4[ch * A0L4Cfg::CAS_LENGTH + col], g0_l4.wts[ch * A0L4Cfg::CAS_LENGTH + col]);
      connect<>(g0_bias4[ch], g0_l4.bias[ch]);
    }
    for (int ch = 0; ch < A0L5Cfg::CAS_NUM; ++ch) {
      for (int col = 0; col < A0L5Cfg::CAS_LENGTH; ++col)
        connect<>(g0_wts5[ch * A0L5Cfg::CAS_LENGTH + col], g0_l5.wts[ch * A0L5Cfg::CAS_LENGTH + col]);
      connect<>(g0_bias5[ch], g0_l5.bias[ch]);
    }
    for (int ch = 0; ch < A0L6Cfg::CAS_NUM; ++ch) {
      for (int col = 0; col < A0L6Cfg::CAS_LENGTH; ++col)
        connect<>(g0_wts6[ch * A0L6Cfg::CAS_LENGTH + col], g0_l6.wts[ch * A0L6Cfg::CAS_LENGTH + col]);
      connect<>(g0_bias6[ch], g0_l6.bias[ch]);
    }
    for (int ch = 0; ch < A0L7Cfg::CAS_NUM; ++ch) {
      for (int col = 0; col < A0L7Cfg::CAS_LENGTH; ++col)
        connect<>(g0_wts7[ch * A0L7Cfg::CAS_LENGTH + col], g0_l7.wts[ch * A0L7Cfg::CAS_LENGTH + col]);
      connect<>(g0_bias7[ch], g0_l7.bias[ch]);
    }

    g0_l0.place_graph(A0L0Cfg::col_placement, A0L0Cfg::row_placement);
    g0_l1.place_graph(A0L1Cfg::col_placement, A0L1Cfg::row_placement);
    g0_l2.place_graph(A0L2Cfg::col_placement, A0L2Cfg::row_placement);
    g0_l3.place_graph(A0L3Cfg::col_placement, A0L3Cfg::row_placement);
    g0_l4.place_graph(A0L4Cfg::col_placement, A0L4Cfg::row_placement);
    g0_l5.place_graph(A0L5Cfg::col_placement, A0L5Cfg::row_placement);
    g0_l6.place_graph(A0L6Cfg::col_placement, A0L6Cfg::row_placement);
    g0_l7.place_graph(A0L7Cfg::col_placement, A0L7Cfg::row_placement);

  }
};

class dut_graph : public graph {
public:
  input_plio  plio_g0_in;
  output_plio plio_g0_out;

  input_port g0_wts0[A0L0Cfg::CAS_NUM * A0L0Cfg::CAS_LENGTH];
  input_port g0_bias0[A0L0Cfg::CAS_NUM];
  input_port g0_wts1[A0L1Cfg::CAS_NUM * A0L1Cfg::CAS_LENGTH];
  input_port g0_bias1[A0L1Cfg::CAS_NUM];
  input_port g0_wts2[A0L2Cfg::CAS_NUM * A0L2Cfg::CAS_LENGTH];
  input_port g0_bias2[A0L2Cfg::CAS_NUM];
  input_port g0_wts3[A0L3Cfg::CAS_NUM * A0L3Cfg::CAS_LENGTH];
  input_port g0_bias3[A0L3Cfg::CAS_NUM];
  input_port g0_wts4[A0L4Cfg::CAS_NUM * A0L4Cfg::CAS_LENGTH];
  input_port g0_bias4[A0L4Cfg::CAS_NUM];
  input_port g0_wts5[A0L5Cfg::CAS_NUM * A0L5Cfg::CAS_LENGTH];
  input_port g0_bias5[A0L5Cfg::CAS_NUM];
  input_port g0_wts6[A0L6Cfg::CAS_NUM * A0L6Cfg::CAS_LENGTH];
  input_port g0_bias6[A0L6Cfg::CAS_NUM];
  input_port g0_wts7[A0L7Cfg::CAS_NUM * A0L7Cfg::CAS_LENGTH];
  input_port g0_bias7[A0L7Cfg::CAS_NUM];

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
    for (int ch = 0; ch < A0L2Cfg::CAS_NUM; ++ch) {
      for (int col = 0; col < A0L2Cfg::CAS_LENGTH; ++col)
        connect<>(g0_wts2[ch * A0L2Cfg::CAS_LENGTH + col], dut.g0_wts2[ch * A0L2Cfg::CAS_LENGTH + col]);
      connect<>(g0_bias2[ch], dut.g0_bias2[ch]);
    }
    for (int ch = 0; ch < A0L3Cfg::CAS_NUM; ++ch) {
      for (int col = 0; col < A0L3Cfg::CAS_LENGTH; ++col)
        connect<>(g0_wts3[ch * A0L3Cfg::CAS_LENGTH + col], dut.g0_wts3[ch * A0L3Cfg::CAS_LENGTH + col]);
      connect<>(g0_bias3[ch], dut.g0_bias3[ch]);
    }
    for (int ch = 0; ch < A0L4Cfg::CAS_NUM; ++ch) {
      for (int col = 0; col < A0L4Cfg::CAS_LENGTH; ++col)
        connect<>(g0_wts4[ch * A0L4Cfg::CAS_LENGTH + col], dut.g0_wts4[ch * A0L4Cfg::CAS_LENGTH + col]);
      connect<>(g0_bias4[ch], dut.g0_bias4[ch]);
    }
    for (int ch = 0; ch < A0L5Cfg::CAS_NUM; ++ch) {
      for (int col = 0; col < A0L5Cfg::CAS_LENGTH; ++col)
        connect<>(g0_wts5[ch * A0L5Cfg::CAS_LENGTH + col], dut.g0_wts5[ch * A0L5Cfg::CAS_LENGTH + col]);
      connect<>(g0_bias5[ch], dut.g0_bias5[ch]);
    }
    for (int ch = 0; ch < A0L6Cfg::CAS_NUM; ++ch) {
      for (int col = 0; col < A0L6Cfg::CAS_LENGTH; ++col)
        connect<>(g0_wts6[ch * A0L6Cfg::CAS_LENGTH + col], dut.g0_wts6[ch * A0L6Cfg::CAS_LENGTH + col]);
      connect<>(g0_bias6[ch], dut.g0_bias6[ch]);
    }
    for (int ch = 0; ch < A0L7Cfg::CAS_NUM; ++ch) {
      for (int col = 0; col < A0L7Cfg::CAS_LENGTH; ++col)
        connect<>(g0_wts7[ch * A0L7Cfg::CAS_LENGTH + col], dut.g0_wts7[ch * A0L7Cfg::CAS_LENGTH + col]);
      connect<>(g0_bias7[ch], dut.g0_bias7[ch]);
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
  for (int ch = 0; ch < A0L2Cfg::CAS_NUM; ++ch) {
    for (int col = 0; col < A0L2Cfg::CAS_LENGTH; ++col) {
      int idx = ch * A0L2Cfg::CAS_LENGTH + col;
      dut.update(dut.g0_wts2[idx], g0_w2[ch][col], A0L2Cfg::IN_FEAT_SLICE * A0L2Cfg::OUT_FEAT_SLICE);
    }
    dut.update(dut.g0_bias2[ch], g0_b2[ch], A0L2Cfg::OUT_FEAT_SLICE);
  }
  for (int ch = 0; ch < A0L3Cfg::CAS_NUM; ++ch) {
    for (int col = 0; col < A0L3Cfg::CAS_LENGTH; ++col) {
      int idx = ch * A0L3Cfg::CAS_LENGTH + col;
      dut.update(dut.g0_wts3[idx], g0_w3[ch][col], A0L3Cfg::IN_FEAT_SLICE * A0L3Cfg::OUT_FEAT_SLICE);
    }
    dut.update(dut.g0_bias3[ch], g0_b3[ch], A0L3Cfg::OUT_FEAT_SLICE);
  }
  for (int ch = 0; ch < A0L4Cfg::CAS_NUM; ++ch) {
    for (int col = 0; col < A0L4Cfg::CAS_LENGTH; ++col) {
      int idx = ch * A0L4Cfg::CAS_LENGTH + col;
      dut.update(dut.g0_wts4[idx], g0_w4[ch][col], A0L4Cfg::IN_FEAT_SLICE * A0L4Cfg::OUT_FEAT_SLICE);
    }
    dut.update(dut.g0_bias4[ch], g0_b4[ch], A0L4Cfg::OUT_FEAT_SLICE);
  }
  for (int ch = 0; ch < A0L5Cfg::CAS_NUM; ++ch) {
    for (int col = 0; col < A0L5Cfg::CAS_LENGTH; ++col) {
      int idx = ch * A0L5Cfg::CAS_LENGTH + col;
      dut.update(dut.g0_wts5[idx], g0_w5[ch][col], A0L5Cfg::IN_FEAT_SLICE * A0L5Cfg::OUT_FEAT_SLICE);
    }
    dut.update(dut.g0_bias5[ch], g0_b5[ch], A0L5Cfg::OUT_FEAT_SLICE);
  }
  for (int ch = 0; ch < A0L6Cfg::CAS_NUM; ++ch) {
    for (int col = 0; col < A0L6Cfg::CAS_LENGTH; ++col) {
      int idx = ch * A0L6Cfg::CAS_LENGTH + col;
      dut.update(dut.g0_wts6[idx], g0_w6[ch][col], A0L6Cfg::IN_FEAT_SLICE * A0L6Cfg::OUT_FEAT_SLICE);
    }
    dut.update(dut.g0_bias6[ch], g0_b6[ch], A0L6Cfg::OUT_FEAT_SLICE);
  }
  for (int ch = 0; ch < A0L7Cfg::CAS_NUM; ++ch) {
    for (int col = 0; col < A0L7Cfg::CAS_LENGTH; ++col) {
      int idx = ch * A0L7Cfg::CAS_LENGTH + col;
      dut.update(dut.g0_wts7[idx], g0_w7[ch][col], A0L7Cfg::IN_FEAT_SLICE * A0L7Cfg::OUT_FEAT_SLICE);
    }
    dut.update(dut.g0_bias7[ch], g0_b7[ch], A0L7Cfg::OUT_FEAT_SLICE);
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
