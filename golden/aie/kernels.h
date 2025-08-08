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
// --- AIE1 Conv2D with K_TILE=8, NHWC * HWIO -> NHWC ---
// K_TOTAL must be divisible by 8
template<
  int H,int W,int CI,
  int KH,int KW,int CO,
  int STRIDE_H,int STRIDE_W,
  int PAD_H,int PAD_W,
  int m,                // output pixels per tile
  int n,                // output channels per tile
  int Tm,               // (HO*WO)/m
  int Tk,               // K_TOTAL/8
  int Tn,               // CO/n
  int K_TOTAL,          // KH*KW*CI
  int SHIFT, bool is_relu
>
void conv2d(
  input_window_int8  * __restrict in_nhwc,   // [H][W][CI]
  output_window_int8 * __restrict out_nhwc,  // [HO][WO][CO]
  const int8         * __restrict matB       // weights tiled as [(K_TOTAL/8),(CO/n),8,n]
){
  using MMUL     = aie::mmul<m, 8, n, int8, int8>;   // <-- K_TILE = 8
  constexpr int K_TILE = 8;

  constexpr int HO = (H + 2*PAD_H - KH)/STRIDE_H + 1;
  constexpr int WO = (W + 2*PAD_W - KW)/STRIDE_W + 1;

  static_assert((HO*WO) % m == 0, "HO*WO must be divisible by m");
  static_assert(CO % n == 0,      "CO   must be divisible by n");
  static_assert(K_TOTAL % K_TILE == 0, "K_TOTAL must be divisible by 8");
  static_assert(Tk == (K_TOTAL / K_TILE), "Tk mismatch");

  const int8* __restrict in  = (const int8*) in_nhwc->ptr;
  int8*       __restrict out = (int8*) out_nhwc->ptr;

  // Local im2col buffer for one MMUL A tile (m x K_TILE)
  alignas(16) int8 A_buf[m * K_TILE];

  // Build one A tile (im2col) into A_buf for slice 'ik'
  auto build_A_tile = [&](int p0, int ik) {
    const int kk0 = ik * K_TILE;
    // Fill as m rows, K_TILE cols
    for (int r = 0; r < m; ++r) {
      int p  = p0 + r;
      int oh = p / WO;
      int ow = p % WO;

      for (int kk = 0; kk < K_TILE; ++kk) {
        int K  = kk0 + kk;               // 0 .. K_TOTAL-1
        int ci =  K % CI;
        int t  =  K / CI;
        int kw =  t % KW;
        int kh =  t / KW;

        int ih = oh*STRIDE_H + kh - PAD_H;
        int iw = ow*STRIDE_W + kw - PAD_W;

        int8 v = 0;
        if ((ih>=0 && ih<H) && (iw>=0 && iw<W)) {
          v = in[((ih*W + iw)*CI) + ci];
        }
        A_buf[r*K_TILE + kk] = v;
      }
    }
  };

  // For profiling
  unsigned long long cycle_num[2];
  aie::tile tile=aie::tile::current();
  cycle_num[0]=tile.cycles();

  for (int im = 0; im < Tm; ++im) chess_unroll_loop(Tm)
  {
    int p0 = im * m; // first output pixel index this tile handles (row-major over HO*WO)

    for (int in_t = 0; in_t < Tn; ++in_t) chess_unroll_loop(Tn)
    {
      // B layout: [(Tk),(Tn), K_TILE*n]. We step Tn outermost.
      const int8* __restrict pB_base = matB + in_t * MMUL::size_B;

      // ik = 0: C = A*B
      build_A_tile(p0, 0);
      aie::vector<int8, MMUL::size_A> A = aie::load_v<MMUL::size_A>(A_buf);
      aie::vector<int8, MMUL::size_B> B = aie::load_v<MMUL::size_B>(pB_base);
      MMUL C; C.mul(A, B);

      // ik = 1..Tk-1: C += A*B
      const int8* __restrict pB_ik = pB_base + (MMUL::size_B * Tn);
      for (int ik = 1; ik < Tk; ++ik) //chssess_flatten_loop
      {
        build_A_tile(p0, ik);
        aie::vector<int8, MMUL::size_A> Aik = aie::load_v<MMUL::size_A>(A_buf);
        aie::vector<int8, MMUL::size_B> Bik = aie::load_v<MMUL::size_B>(pB_ik); 
        pB_ik += (MMUL::size_B * Tn);
        C.mac(Aik, Bik);
      }

      // Quant + (optional) ReLU
      auto C_vec = C.template to_vector<int8>(SHIFT);
      auto C_out = is_relu ? aie::max(C_vec, (int8)0) : C_vec;

      // Scatter to NHWC
      for (int r = 0; r < m; ++r) {
        int p  = p0 + r;
        int oh = p / WO;
        int ow = p % WO;
        int co_base = in_t * n;
        int out_base = ((oh*WO + ow) * CO) + co_base;
        for (int c = 0; c < n; ++c) {
          out[out_base + c] = C_out[r*n + c];
        }
      }
    }
  }

  cycle_num[1]=tile.cycles();
  printf("conv2d start=%lld,end=%lld,total=%lld\n",cycle_num[0],cycle_num[1],cycle_num[1]-cycle_num[0]);
}

// NHWC -> (H*W, C) flatten; bytewise copy in layout order
template<int H, int W, int C>
void flatten_nhwc_to_hw_by_c(
  input_window_int8  * __restrict in_nhwc,
  output_window_int8 * __restrict out_hw_by_c
){
  const int total = H * W * C;
  const int8* __restrict src = (const int8*) in_nhwc->ptr;
  int8*       __restrict dst = (int8*)       out_hw_by_c->ptr;
  for (int i = 0; i < total; ++i) dst[i] = src[i];
}

#endif