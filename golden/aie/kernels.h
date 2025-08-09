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

// Pack two rows (top then bottom) of 8 CI-lanes into a 16B vector
template<int H, int W, int CI>
static inline aie::vector<int8,16>
load_pack16_tb(const int8* __restrict in, int ih, int iw, int ci8,
               const int8* __restrict ZERO8) __attribute__((always_inline)) {
  alignas(32) int8 tmp[16];
  const bool in0 = (ih >= 0 && ih < H) && (iw >= 0 && iw < W);          // top
  const bool in1 = (ih+1 >= 0 && ih+1 < H) && (iw >= 0 && iw < W);      // bottom
  const int8* p0 = in0 ? &in[((ih*W + iw)*CI) + ci8*8]     : ZERO8;     // top
  const int8* p1 = in1 ? &in[(((ih+1)*W + iw)*CI) + ci8*8] : ZERO8;     // bottom
  __builtin_memcpy(&tmp[0],  p0, 8);
  __builtin_memcpy(&tmp[8],  p1, 8);
  return aie::load_v<16>(tmp);
}

// Weights expected as [CI8][Tn][3][3][lane=8][n=8]  (64 bytes per tap)
template<int H, int W, int CI, int CO, int PAD_H, int PAD_W>
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

  using MMUL = aie::mmul<2,8,8,int8,int8>;
  constexpr int KH = 3, KW = 3;
  constexpr int HO = (H + 2*PAD_H - KH) + 1;     // stride=1
  constexpr int WO = (W + 2*PAD_W - KW) + 1;
  static_assert((HO % 2) == 0, "HO must be even for m=2");

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
    const int oh_pair = (im / WO) * 2;     // 0,2,4,...
    const int ow      = (im % WO);

    for (int itn = 0; itn < Tn; ++itn) {
      MMUL C;

      // Accumulate over all taps (ci8 × kh × kw)
      bool first = true;
      for (int ci8 = 0; ci8 < CI8; ++ci8) {
        const int8* __restrict pBci = matB + ci8*B_stride_ci8 + itn*B_stride_tn;

        for (int kh = 0; kh < KH; ++kh) {
          const int ih = (oh_pair - PAD_H) + kh;
          for (int kw = 0; kw < KW; ++kw) {
            const int iw = (ow - PAD_W) + kw;

            auto Av = load_pack16_tb<H,W,CI>(in, ih, iw, ci8, ZERO8);
            auto Bv = aie::load_v<64>(pBci);   pBci += TAP_BYTES;

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
      const int o0 = ((oh_pair  )*WO + ow) * CO + co_base;
      const int o1 = ((oh_pair+1)*WO + ow) * CO + co_base;
      __builtin_memcpy(&out[o0], &cpack[0],  8);
      __builtin_memcpy(&out[o1], &cpack[8],  8);
    }
  }

  unsigned long long c1 = t.cycles();
  printf("conv2d_v_tiny cycles=%llu (H=%d W=%d CI=%d CO=%d PAD=(%d,%d))\n",
         (unsigned long long)(c1 - c0), H, W, CI, CO, PAD_H, PAD_W);
}


#endif