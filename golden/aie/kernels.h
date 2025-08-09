#ifndef FUNCTION_KERNELS_H
#define FUNCTION_KERNELS_H

#include <adf.h>
#include "aie_api/aie.hpp"
#include "aie_api/aie_adf.hpp"

// -----------------------------------------------------------------------------
// Existing DENSE (unchanged)
// -----------------------------------------------------------------------------
template <int m, int k, int n, int Tm, int Tk, int Tn, int SHIFT, bool is_relu>
void dense(
  input_window_int8 * __restrict matA,
  output_window_int8 * __restrict matC,
  const int8 matB []
){
  using MMUL = aie::mmul<m, k, n, int8, int8>;

  const int8* __restrict pA = (int8*)matA->ptr;
  const int8* __restrict pB = (int8*)matB;
  int8*       __restrict pC = (int8*)matC->ptr;

  // For profiling only
  unsigned long long cycle_num[2];
  aie::tile tile = aie::tile::current();
  cycle_num[0] = tile.cycles();

  for (unsigned im = 0; im < Tm; ++im) chess_unroll_loop(Tm)
  {
    for (unsigned in = 0; in < Tn; ++in) chess_unroll_loop(Tn)
    {
      const int8 * __restrict pA1 = pA + (im * Tk + 0) * MMUL::size_A;
      const int8 * __restrict pB1 = pB + (0  * Tn + in) * MMUL::size_B;

      aie::vector<int8, MMUL::size_A> A = aie::load_v<MMUL::size_A>(pA1); pA1 += MMUL::size_A;
      aie::vector<int8, MMUL::size_B> B = aie::load_v<MMUL::size_B>(pB1); pB1 += MMUL::size_B * Tn;

      MMUL C; C.mul(A, B);

      for (unsigned ik = 0; ik < Tk-1; ++ik) chess_flatten_loop
      {
        A = aie::load_v<MMUL::size_A>(pA1); pA1 += MMUL::size_A;
        B = aie::load_v<MMUL::size_B>(pB1); pB1 += MMUL::size_B * Tn;
        C.mac(A, B);
      }

      auto C_vec = C.template to_vector<int8>(SHIFT);
      auto C_out = is_relu ? aie::max(C_vec, (int8)0) : C_vec;
      aie::store_v(pC, C_out); pC += MMUL::size_C;
    }
  }

  cycle_num[1] = tile.cycles();
  printf("dense start=%lld,end=%lld,total=%lld\n", cycle_num[0], cycle_num[1], cycle_num[1]-cycle_num[0]);
}



enum Padding { VALID, SAME };

// pack top & bottom rows (separated by SH) for one XC8 block into 16 bytes.
// order: [top(8), bottom(8)]  — matches your working kernel
template<int XH, int XW, int XC, int SH>
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

// Compute padding for SAME or VALID
consteval int compute_pad(const int in_size, const int k, const int stride, const Padding mode) {
  if (mode == VALID) return 0;
  const int out_size  = (in_size + stride - 1) / stride; // ceil(in/stride)
  const int total_pad = ((out_size - 1) * stride + k - in_size);
  return total_pad < 0 ? 0 : total_pad / 2;              // symmetric
}

// Weights expected as [XC8][YC8][KH][KW][lane=8][n=8] (64 bytes per tap)
template<int XH, int XW, int XC, int YC,
         int KH, int KW,
         int SH, int SW,
         Padding PMode>
void conv2d_v_tiny(
    input_window_int8  * __restrict in_nhwc,
    output_window_int8 * __restrict out_nhwc,
    const int8         * __restrict matB,
    const int SHIFT,
    const bool DO_RELU)
{
  static_assert((XC % 8) == 0 && (YC % 8) == 0, "XC & YC must be multiples of 8");
  static_assert(SH >= 1 && SW >= 1, "Strides must be >= 1");

  constexpr int PAD_H = compute_pad(XH, KH, SH, PMode);
  constexpr int PAD_W = compute_pad(XW, KW, SW, PMode);

  using MMUL = aie::mmul<2,8,8,int8,int8>; // m=2,k=8,n=8

  constexpr int YH = ((XH + 2*PAD_H - KH) / SH) + 1;
  constexpr int YW = ((XW + 2*PAD_W - KW) / SW) + 1;

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
  unsigned long long c0 = t.cycles();

  // ---- Single loop over row-pairs, with optional tail handling in-place ----
  for (int yh2 = 0; yh2 < YH; yh2 += 2) {
    for (int yw = 0; yw < YW; ++yw) {
      for (int yc8 = 0; yc8 < YC8; ++yc8) {
        MMUL C; bool first = true;

        for (int xc8 = 0; xc8 < XC8; ++xc8) {
          const int8* __restrict pBci = matB + xc8*B_stride_xc8 + yc8*B_stride_yc8;

          for (int kh = 0; kh < KH; ++kh)
          {
            const int xh_top = yh2*SH - PAD_H + kh;

            for (int kw = 0; kw < KW; ++kw) {
              const int xw = yw*SW - PAD_W + kw;

              // pack [top, bottom]; if bottom row is beyond YH, we’ll still compute it
              // but simply won’t store it (top is always correct).
              auto Av = pack_tb16<XH,XW,XC,SH>(in, xh_top, xw, xc8, ZERO8);
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
  
  unsigned long long c1 = t.cycles();
  unsigned long long cycles = c1 - c0;
  constexpr int macs = XH * XW * XC * KH * KW * YC;
  constexpr unsigned long long cycles_expected = macs / 128;
  double efficiency = 100* (double)cycles_expected / cycles;
  printf("\n\n-----------conv2d_v_tiny efficiency=(%.1f%%), cycles=%llu, cycles_expected=%llu (XH=%d XW=%d XC=%d YC=%d KH=%d KW=%d PAD=(%d,%d) STRIDE=(%d,%d) PMode=%s)\n",
         efficiency, cycles, cycles_expected, XH, XW, XC, YC, KH, KW,
         PAD_H, PAD_W, SH, SW,
         (PMode == VALID ? "VALID" : "SAME"));
}




#endif