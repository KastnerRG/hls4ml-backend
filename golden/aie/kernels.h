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



// Weights expected as [CI8][Tn][KH][KW][lane=8][n=8]  (64 bytes per tap)
template<int H, int W, int CI, int CO,
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
  static_assert((CI % 8) == 0 && (CO % 8) == 0, "CI & CO must be multiples of 8");
  static_assert(PAD_H == 0 || PAD_H == 1, "PAD_H ∈ {0,1}");
  static_assert(PAD_W == 0 || PAD_W == 1, "PAD_W ∈ {0,1}");
  static_assert(SH >= 1 && SW >= 1, "Strides must be >= 1");

  using MMUL = aie::mmul<2,8,8,int8,int8>;         // m=2, k=8, n=8

  constexpr int HO = ((H + 2*PAD_H - KH) / SH) + 1;
  constexpr int WO = ((W + 2*PAD_W - KW) / SW) + 1;
  static_assert((HO % 2) == 0, "HO must be even for m=2 kernel");

  const int CI8 = CI / 8;
  const int Tn  = CO / 8;

  const int8* __restrict in  = (const int8*)in_nhwc->ptr;
  int8*       __restrict out = (int8*)      out_nhwc->ptr;

  // [CI8][Tn][KH][KW][lane=8][n=8] -> contiguous 64B per tap
  constexpr int TAP_BYTES  = 64;               // 8 lanes × 8 n
  const int B_stride_tn    = KH * KW * TAP_BYTES;
  const int B_stride_ci8   = Tn * B_stride_tn;

  alignas(32) static const int8 ZERO8[8] = {0};

  // iterate row-pairs then columns: exactly (HO/2)*WO tiles of m=2
  const int OH2 = HO / 2;
  const int Tm  = OH2 * WO;

  aie::tile t = aie::tile::current();
  unsigned long long c0 = t.cycles();

  for (int im = 0; im < Tm; ++im) {
    const int oh_pair = (im / WO) * 2;      // output row index: 0,2,4,...
    const int ow      = (im % WO);

    // Spatial -> input coordinates base (top row of the pair)
    const int base_oh = oh_pair;
    const int base_ow = ow;

    for (int itn = 0; itn < Tn; ++itn) {
      MMUL C;
      bool first = true;

      // Accumulate over all ci8 × KH × KW taps
      for (int ci8 = 0; ci8 < CI8; ++ci8) {
        const int8* __restrict pBci = matB + ci8*B_stride_ci8 + itn*B_stride_tn;

        for (int kh = 0; kh < KH; ++kh) {
          // Input rows corresponding to the two output rows (top/bottom) for this kh
          const int ih_top = base_oh*SH - PAD_H + kh;
          for (int kw = 0; kw < KW; ++kw) {
            const int iw = base_ow*SW - PAD_W + kw;

            // load_pack16_tb_strided
            // Pack top & bottom rows (separated by SH) for one ci8 block into 16 bytes.
            // Order: top first 8 bytes, bottom next 8 bytes (matches your working kernel).
            alignas(32) int8 load_packed[16];
            const bool in_top = (ih_top >= 0 && ih_top < H) && (iw >= 0 && iw < W);
            const int  ih_bot = ih_top + SH;
            const bool in_bot = (ih_bot >= 0 && ih_bot < H) && (iw >= 0 && iw < W);

            const int8* p_top = in_top ? &in[((ih_top*W + iw)*CI) + ci8*8] : ZERO8;
            const int8* p_bot = in_bot ? &in[((ih_bot*W + iw)*CI) + ci8*8] : ZERO8;

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

      // Write two rows (top 8 bytes, bottom 8 bytes)
      alignas(32) int8 cpack[16];
      aie::store_v(cpack, Co);

      const int co_base = itn * 8;
      const int o0 = ((oh_pair  )*WO + ow) * CO + co_base;  // top row of pair
      const int o1 = ((oh_pair+1)*WO + ow) * CO + co_base;  // bottom row of pair
      __builtin_memcpy(&out[o0], &cpack[0],  8);
      __builtin_memcpy(&out[o1], &cpack[8],  8);
    }
  }

  unsigned long long c1 = t.cycles();
  printf("conv2d_v_tiny cycles=%llu (H=%d W=%d CI=%d CO=%d KH=%d KW=%d PAD=(%d,%d) STRIDE=(%d,%d))\n",
         (unsigned long long)(c1 - c0), H, W, CI, CO, KH, KW, PAD_H, PAD_W, SH, SW);
}



#endif