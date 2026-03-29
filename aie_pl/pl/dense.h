#pragma once
#include <ap_int.h>

namespace nnet {

// block_factor = N_IN * N_OUT / RF  (MACs per reuse pipeline stage)
// chunk        = N_IN / RF          (input elements processed per stage)
// Constraint: RF must divide N_IN evenly.
template<int N_IN, int N_OUT, int REUSE_FACTOR>
struct dense_config {
    static const unsigned n_in          = N_IN;
    static const unsigned n_out         = N_OUT;
    static const unsigned reuse_factor  = REUSE_FACTOR;
    static const unsigned block_factor  = N_IN * N_OUT / REUSE_FACTOR;
    static const unsigned chunk         = N_IN / REUSE_FACTOR;
};

// ── hls4ml resource strategy ─────────────────────────────────────────────────
// Structure mirrors dense_resource_rf_leq_nin from nnet_dense_resource.h:
//   ReuseLoop (RF iterations, PIPELINE II=1)
//     └─ ChunkLoop × OutLoop (block_factor MACs, all UNROLL'd)
//
// ARRAY_RESHAPE block keeps weights in BRAM (block_factor elements per row,
// one BRAM read per ReuseLoop iteration).  acc[] stays in registers.
// HLS naturally maps the int8×int8 MACs to DSP48E2 slices.
// ─────────────────────────────────────────────────────────────────────────────
template<typename Config>
void dense_relu(
    ap_int<8>  in[Config::n_in],
    ap_int<8>  out[Config::n_out],
    ap_int<8>  weights[Config::n_in * Config::n_out],
    ap_int<32> biases[Config::n_out],
    int        shift)
{
    #pragma HLS INLINE

    // block_factor consecutive weights per ReuseLoop iteration → one BRAM read.
    const int block_factor = Config::block_factor;
    #pragma HLS ARRAY_RESHAPE   variable=weights block factor=block_factor
    #pragma HLS ARRAY_PARTITION variable=biases  complete

    ap_int<32> acc[Config::n_out];
    #pragma HLS ARRAY_PARTITION variable=acc complete

    for (int j = 0; j < Config::n_out; j++) {
        #pragma HLS UNROLL
        acc[j] = biases[j];
    }

    for (int ir = 0; ir < Config::reuse_factor; ir++) {
        #pragma HLS PIPELINE II=1 rewind
        for (int ic = 0; ic < Config::chunk; ic++) {
            #pragma HLS UNROLL
            int i = ir * Config::chunk + ic;
            ap_int<8> x = in[i];
            for (int j = 0; j < Config::n_out; j++) {
                #pragma HLS UNROLL
                acc[j] += (ap_int<32>)x * (ap_int<32>)weights[i * Config::n_out + j];
            }
        }
    }

    for (int j = 0; j < Config::n_out; j++) {
        #pragma HLS UNROLL
        ap_int<32> shifted = acc[j] >> shift;
        out[j] = (ap_int<8>)((shifted > 127) ? ap_int<32>(127) : (shifted > 0) ? shifted : ap_int<32>(0));
    }
}

// Unsigned-input variant: used when the preceding layer is AIE (relu output → uint8).
template<typename Config>
void dense_relu_u8(
    ap_uint<8> in[Config::n_in],
    ap_int<8>  out[Config::n_out],
    ap_int<8>  weights[Config::n_in * Config::n_out],
    ap_int<32> biases[Config::n_out],
    int        shift)
{
    #pragma HLS INLINE

    const int block_factor = Config::block_factor;
    #pragma HLS ARRAY_RESHAPE   variable=weights block factor=block_factor
    #pragma HLS ARRAY_PARTITION variable=biases  complete

    ap_int<32> acc[Config::n_out];
    #pragma HLS ARRAY_PARTITION variable=acc complete

    for (int j = 0; j < Config::n_out; j++) {
        #pragma HLS UNROLL
        acc[j] = biases[j];
    }

    for (int ir = 0; ir < Config::reuse_factor; ir++) {
        #pragma HLS PIPELINE II=1 rewind
        for (int ic = 0; ic < Config::chunk; ic++) {
            #pragma HLS UNROLL
            int i = ir * Config::chunk + ic;
            ap_int<32> x = (ap_int<32>)(ap_uint<8>)in[i];
            for (int j = 0; j < Config::n_out; j++) {
                #pragma HLS UNROLL
                acc[j] += x * (ap_int<32>)weights[i * Config::n_out + j];
            }
        }
    }

    for (int j = 0; j < Config::n_out; j++) {
        #pragma HLS UNROLL
        ap_int<32> shifted = acc[j] >> shift;
        out[j] = (ap_int<8>)((shifted > 127) ? ap_int<32>(127) : (shifted > 0) ? shifted : ap_int<32>(0));
    }
}

} // namespace nnet
