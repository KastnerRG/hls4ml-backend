
#include <adf.h>
#include "aie_api/aie.hpp"
#include "aie_api/aie_adf.hpp"

// pack top & bottom rows (separated by SH) for one XC8 block into 16 bytes.
// order: [top(8), bottom(8)]  — matches your working kernel
static inline aie::vector<int8,16>
pack_tb16(const int8* __restrict in, int xh_top, int xw, int xc8,
          const int8* __restrict ZERO8)
{
  alignas(32) int8 buf[16];
  const bool in_top = (xh_top >= 0 && xh_top < XH) && (xw >= 0 && xw < XW);
  const int  xh_bot = xh_top + SH;
  const bool in_bot = (xh_bot >= 0 && xh_bot < XH) && (xw >= 0 && xw < XW);

  const int8* p_top = in_top ? &in[((xh_top*XW + xw)*XC) + xc8*8] : ZERO8;
  const int8* p_bot = in_bot ? &in[((xh_bot*XW + xw)*XC) + xc8*8] : ZERO8;

  __builtin_memcpy(&buf[0], p_top, 8);
  __builtin_memcpy(&buf[8], p_bot, 8);
  return aie::load_v<16>(buf);
}

// Weights expected as [XC8][YC8][KH][KW][lane=8][n=8] (64 bytes per tap)
void conv2d_i8(
    input_window_int8  * __restrict in_nhwc,
    output_window_int8 * __restrict out_nhwc)
{
  static_assert((XC % 8) == 0 && (YC % 8) == 0, "XC & YC must be multiples of 8");
  static_assert(SH >= 1 && SW >= 1, "Strides must be >= 1");

  using MMUL = aie::mmul<2,8,8,int8,int8>; // m=2,k=8,n=8

  constexpr int XC8 = XC / 8;
  constexpr int YC8 = YC / 8; // Tn

  const int8* __restrict in  = (const int8*)in_nhwc->ptr;
  int8*       __restrict out = (int8*)      out_nhwc->ptr;

  // [XC8][YC8][KH][KW][8][8] => 64 bytes per tap
  constexpr int TAP_BYTES    = 64;
  constexpr int B_stride_yc8 = KH * KW * TAP_BYTES;
  constexpr int B_stride_xc8 = YC8 * B_stride_yc8;

  alignas(32) static const int8 ZERO8[8] = {0};

  aie::tile t = aie::tile::current();
  uint64 c0 = t.cycles();

  // ---- Single loop over row-pairs, with optional tail handling in-place ----
  for (int yh2 = 0; yh2 < YH; yh2 += 2) {
    for (int yw = 0; yw < YW; ++yw) {
      for (int yc8 = 0; yc8 < YC8; ++yc8) {
        MMUL C; bool first = true;

        for (int xc8 = 0; xc8 < XC8; ++xc8) {
          const int8* __restrict pBci = k_p + xc8*B_stride_xc8 + yc8*B_stride_yc8;

          for (int kh = 0; kh < KH; ++kh)
          {
            const int xh_top = yh2*SH - PH + kh;

            for (int kw = 0; kw < KW; ++kw) chess_unroll_loop(KW)
            {
              const int xw = yw*SW - PW + kw;

              // pack [top, bottom]; if bottom row is beyond YH, we’ll still compute it
              // but simply won’t store it (top is always correct).
              auto Av = pack_tb16(in, xh_top, xw, xc8, ZERO8);
              auto Bv = aie::load_v<64>(pBci);  pBci += TAP_BYTES;

              if (first) { C.mul(Av, Bv); first = false; }
              else       { C.mac(Av, Bv); }
            }
          }
        }

        // Quantize + (optional) ReLU
        auto Cv = C.template to_vector<int8>(SHIFT);
        auto Co = DO_RELU ? aie::max(Cv, (int8)0) : Cv;

        // Store: always write the top row; write bottom row only if it exists
        alignas(32) int8 cpack[16];
        aie::store_v(cpack, Co);

        const int yc_base = yc8 * 8;
        const int o_top = ((yh2  )*YW + yw) * YC + yc_base;
        __builtin_memcpy(&out[o_top], &cpack[0], 8);

        const bool have_bottom = (YH % 2 == 0) || (yh2 + 1 < YH);
        if (have_bottom) {
          const int o_bot = ((yh2+1)*YW + yw) * YC + yc_base;
          __builtin_memcpy(&out[o_bot], &cpack[8], 8);
        }
      }
    }
  }
  
  uint64 c1 = t.cycles();
  uint64 cycles = c1 - c0;
  uint64 macs = (uint64)XH * (uint64)XW * (uint64)XC * (uint64)KH * (uint64)KW * (uint64)YC;
  uint64 cycles_expected = macs / 128;
  double efficiency = 100* (double)cycles_expected / cycles;
  printf("\n\n-----------conv2d_i8 efficiency=(%.1f%%), cycles=%llu, cycles_expected=%llu (XH=%d XW=%d XC=%d YC=%d KH=%d KW=%d PAD=(%d,%d) STRIDE=(%d,%d))\n",
         efficiency, cycles, cycles_expected, XH, XW, XC, YC, KH, KW,
         PH, PW, SH, SW);
}