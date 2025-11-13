#include <adf.h>
#include <cstdint>

#ifndef SHIFT
#define SHIFT 0
#endif
#ifndef DO_RELU
#define DO_RELU false
#endif

#ifndef XH
#error "XH not defined"
#endif

static inline void conv_window_simple(
    input_window_int8* __restrict win_in,
    output_window_int8* __restrict win_out)
{
  const int8* __restrict in = (const int8*)win_in->ptr;
  int8* __restrict out = (int8*)win_out->ptr;

  constexpr int IN_STRIDE = XW * CI;
  constexpr int OUT_STRIDE = YW * CO;

  for (int oh = 0; oh < YH; ++oh) {
    for (int ow = 0; ow < YW; ++ow) {
      for (int oc = 0; oc < CO; ++oc) {
        int32 acc = 0;
        for (int kh = 0; kh < KH; ++kh) {
          const int ih = oh*SH - PH + kh;
          if (ih < 0 || ih >= XH) continue;
          for (int kw = 0; kw < KW; ++kw) {
            const int iw = ow*SW - PW + kw;
            if (iw < 0 || iw >= XW) continue;
            const int8* __restrict in_ptr = in + (ih*XW + iw)*CI;
            const int8* __restrict wt_ptr = k_p + ((kh*KW + kw)*CI*CO) + oc;
            for (int ic = 0; ic < CI; ++ic) {
              acc += (int32)in_ptr[ic] * (int32)wt_ptr[ic*CO];
            }
          }
        }
        if (SHIFT > 0) {
          acc = (acc + (1 << (SHIFT - 1))) >> SHIFT;
        }
        if (DO_RELU && acc < 0) acc = 0;
        if (acc > 127) acc = 127;
        if (acc < -128) acc = -128;
        out[(oh*YW + ow)*CO + oc] = (int8)acc;
      }
    }
  }
}
