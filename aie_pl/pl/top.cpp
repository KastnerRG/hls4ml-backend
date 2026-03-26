#include "dense.h"

static const int BATCH = 8;
static const int FEAT  = 64;
static const int SHIFT = 0;

using Cfg = nnet::dense_config<FEAT, FEAT, 1>;

void pl_model(
    ap_int<8>  in[BATCH * FEAT],
    ap_int<8>  out[BATCH * FEAT],
    ap_int<8>  w1[FEAT * FEAT],
    ap_int<32> b1[FEAT],
    ap_int<8>  w2[FEAT * FEAT],
    ap_int<32> b2[FEAT])
{
    #pragma HLS INTERFACE ap_memory port=in
    #pragma HLS INTERFACE ap_memory port=out

    ap_int<8> mid[BATCH * FEAT];

Layer1:
    for (int i = 0; i < BATCH; i++)
        nnet::dense_relu<Cfg>(in + i * FEAT, mid + i * FEAT, w1, b1, SHIFT);

Layer2:
    for (int i = 0; i < BATCH; i++)
        nnet::dense_relu<Cfg>(mid + i * FEAT, out + i * FEAT, w2, b2, SHIFT);
}
