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



// Weights expected as [XC8][YC8][KH][KW][lane=8][n=8]  (64 bytes per tap)
template<int XH, int XW, int XC, int YC,
         int KH, int KW,
         int PAD_H, int PAD_W,
         int SH, int SW>
void conv2d_v_tiny(
    input_window_int8  * __restrict in_nhwc,
    output_window_int8 * __restrict out_nhwc,
    const int8         * __restrict matB,
    const int SHIFT,
    const bool DO_RELU)
{
  static_assert((XC % 8) == 0 && (YC % 8) == 0, "XC & YC must be multiples of 8");
  static_assert(PAD_H == 0 || PAD_H == 1, "PAD_H ∈ {0,1}");
  static_assert(PAD_W == 0 || PAD_W == 1, "PAD_W ∈ {0,1}");
  static_assert(SH >= 1 && SW >= 1, "Strides must be >= 1");
  
  using MMUL = aie::mmul<2,8,8,int8,int8>;         // m=2, k=8, n=8
  
  constexpr int YH = ((XH + 2*PAD_H - KH) / SH) + 1;
  constexpr int YW = ((XW + 2*PAD_W - KW) / SW) + 1;
  static_assert((YH % 2) == 0, "YH must be even for m=2 kernel");

  const int XC8 = XC / 8;
  const int YC8 = YC / 8; // = Tn

  const int8* __restrict in  = (const int8*)in_nhwc->ptr;
  int8*       __restrict out = (int8*)      out_nhwc->ptr;

  // [XC8][YC8][KH][KW][lane=8][n=8] -> contiguous 64B per tap
  constexpr int TAP_BYTES  = 64;               // 8 lanes × 8 n
  const int B_stride_yc8   = KH * KW * TAP_BYTES;
  const int B_stride_xc8   = YC8 * B_stride_yc8;

  alignas(32) static const int8 ZERO8[8] = {0};

  aie::tile t = aie::tile::current();
  unsigned long long c0 = t.cycles();

  for (int i_yh_x2 = 0; i_yh_x2 < YH; i_yh_x2 += 2) { // iterate over output YH pairs
    for (int i_yw = 0; i_yw < YW; ++i_yw) {

      // Spatial -> input coordinates base (top YH of the pair)
      const int base_yh = i_yh_x2;
      const int base_yw = i_yw;

      for (int i_yc8 = 0; i_yc8 < YC8; ++i_yc8) { // over n sized tiles (Tn)
        MMUL C;
        bool first = true;

        // Accumulate over all i_xc8 × KH × KW taps
        for (int i_xc8 = 0; i_xc8 < XC8; ++i_xc8) {
          const int8* __restrict pBci = matB + i_xc8*B_stride_xc8 + i_yc8*B_stride_yc8;

          for (int kh = 0; kh < KH; ++kh) {
            // Input ri_yws corresponding to the two output ri_yws (top/bottom) for this kh
            const int i_xh_top = base_yh*SH - PAD_H + kh;
            for (int kw = 0; kw < KW; ++kw) {
              const int i_xw = base_yw*SW - PAD_W + kw;

              // load_pack16_tb_strided
              // Pack top & bottom ri_yws (separated by SH) for one i_xc8 block into 16 bytes.
              // Order: top first 8 bytes, bottom next 8 bytes (matches your working kernel).
              alignas(32) int8 load_packed[16];
              const bool in_top = (i_xh_top >= 0 && i_xh_top < XH) && (i_xw >= 0 && i_xw < XW);
              const int  i_xh_bot = i_xh_top + SH;
              const bool in_bot = (i_xh_bot >= 0 && i_xh_bot < XH) && (i_xw >= 0 && i_xw < XW);

              const int8* p_top = in_top ? &in[((i_xh_top*XW + i_xw)*XC) + i_xc8*8] : ZERO8;
              const int8* p_bot = in_bot ? &in[((i_xh_bot*XW + i_xw)*XC) + i_xc8*8] : ZERO8;

              __builtin_memcpy(&load_packed[0],  p_top, 8);
              __builtin_memcpy(&load_packed[8],  p_bot, 8);

              auto Av = aie::load_v<16>(load_packed);
              auto Bv = aie::load_v<64>(pBci);  pBci += TAP_BYTES;

              if (first) { C.mul(Av, Bv); first = false; }
              else       { C.mac(Av, Bv); }
            }
          }
        }

        // Quantize and (optional) ReLU
        auto Cv = C.template to_vector<int8>(SHIFT);
        auto Co = DO_RELU ? aie::max(Cv, (int8)0) : Cv;

        // Write two ri_yws (top 8 bytes, bottom 8 bytes)
        alignas(32) int8 cpack[16];
        aie::store_v(cpack, Co);

        const int co_base = i_yc8 * 8;
        const int o0 = ((i_yh_x2  )*YW + i_yw) * YC + co_base;  // top ri_yw of pair
        const int o1 = ((i_yh_x2+1)*YW + i_yw) * YC + co_base;  // bottom ri_yw of pair
        __builtin_memcpy(&out[o0], &cpack[0],  8);
        __builtin_memcpy(&out[o1], &cpack[8],  8);
      }
    }
  }

  unsigned long long c1 = t.cycles();
  printf("conv2d_v_tiny cycles=%llu (XH=%d XW=%d XC=%d YC=%d KH=%d KW=%d PAD=(%d,%d) STRIDE=(%d,%d))\n",
         (unsigned long long)(c1 - c0), XH, XW, XC, YC, KH, KW, PAD_H, PAD_W, SH, SW);
}



#endif