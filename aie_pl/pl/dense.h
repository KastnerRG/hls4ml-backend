#pragma once
#include <ap_int.h>

namespace nnet {

template<int N_IN, int N_OUT, int REUSE_FACTOR>
struct dense_config {
    static const unsigned n_in  = N_IN;
    static const unsigned n_out = N_OUT;
    static const unsigned reuse_factor = REUSE_FACTOR;
};

template<typename Config>
void dense_relu(
    ap_int<8>  in[Config::n_in],
    ap_int<8>  out[Config::n_out],
    ap_int<8>  weights[Config::n_in * Config::n_out],
    ap_int<32> biases[Config::n_out],
    int        shift)
{
    #pragma HLS PIPELINE II=Config::reuse_factor
    #pragma HLS ARRAY_PARTITION variable=out complete

    ap_int<32> acc[Config::n_out];
    #pragma HLS ARRAY_PARTITION variable=acc complete

Product1:
    for (int j = 0; j < Config::n_out; j++)
        acc[j] = biases[j];

Product2:
    for (int i = 0; i < Config::n_in; i++) {
        ap_int<8> cache = in[i];
    Product3:
        for (int j = 0; j < Config::n_out; j++)
            acc[j] += (ap_int<32>)cache * weights[i * Config::n_out + j];
    }

Result:
    for (int j = 0; j < Config::n_out; j++) {
        ap_int<32> shifted = acc[j] >> shift;
        out[j] = (ap_int<8>)((shifted > 127) ? ap_int<32>(127) : (shifted > 0) ? shifted : ap_int<32>(0));
    }
}

// Unsigned-input variant: used when the preceding layer is AIE (which outputs
// uint8 relu values in [0,255]).  Accumulator treats input as unsigned so that
// values > 127 are not sign-extended — matching the numpy uint8 reference.
template<typename Config>
void dense_relu_u8(
    ap_uint<8> in[Config::n_in],
    ap_int<8>  out[Config::n_out],
    ap_int<8>  weights[Config::n_in * Config::n_out],
    ap_int<32> biases[Config::n_out],
    int        shift)
{
    #pragma HLS PIPELINE II=Config::reuse_factor
    #pragma HLS ARRAY_PARTITION variable=out complete

    ap_int<32> acc[Config::n_out];
    #pragma HLS ARRAY_PARTITION variable=acc complete

Product1_u8:
    for (int j = 0; j < Config::n_out; j++)
        acc[j] = biases[j];

Product2_u8:
    for (int i = 0; i < Config::n_in; i++) {
        ap_uint<8> cache = in[i];
    Product3_u8:
        for (int j = 0; j < Config::n_out; j++)
            acc[j] += (ap_int<32>)cache * (ap_int<32>)weights[i * Config::n_out + j];
    }

Result_u8:
    for (int j = 0; j < Config::n_out; j++) {
        ap_int<32> shifted = acc[j] >> shift;
        out[j] = (ap_int<8>)((shifted > 127) ? ap_int<32>(127) : (shifted > 0) ? shifted : ap_int<32>(0));
    }
}

} // namespace nnet
