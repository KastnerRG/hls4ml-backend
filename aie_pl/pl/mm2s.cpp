// Memory-mapped to AXI4-Stream DMA mover.
// Reads `size` 32-bit words from DDR and streams them out.
#include <ap_int.h>
#include <hls_stream.h>
#include <ap_axi_sdata.h>

extern "C" {
void mm2s(ap_int<32>* mem, hls::stream<ap_axis<32,0,0,0>>& s, int size) {
#pragma HLS INTERFACE m_axi     port=mem  offset=slave bundle=gmem
#pragma HLS INTERFACE axis      port=s
#pragma HLS INTERFACE s_axilite port=mem   bundle=control
#pragma HLS INTERFACE s_axilite port=size  bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    for (int i = 0; i < size; i++) {
#pragma HLS PIPELINE II=1
        ap_axis<32,0,0,0> x;
        x.data = mem[i];
        x.keep = -1; x.strb = -1;
        x.last = (i == size - 1) ? 1 : 0;
        s.write(x);
    }
}
}
