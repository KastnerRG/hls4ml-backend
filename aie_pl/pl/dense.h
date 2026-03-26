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
        out[j] = (shifted > 0) ? (ap_int<8>)(shifted > 127 ? 127 : shifted) : ap_int<8>(0);
    }
}

} // namespace nnet
