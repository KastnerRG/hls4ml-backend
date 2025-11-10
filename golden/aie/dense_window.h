// dense_i8.h
#include <adf.h>
#include "aie_api/aie.hpp"
#include "aie_api/aie_adf.hpp"

#define CAT(a,b) a##b
#define XCAT(a,b) CAT(a,b)

#define Tm (mm_M / mm_m)
#define Tk (mm_K / mm_k)
#define Tn (mm_N / mm_n)

// Optional: enforce that we really can unroll by 2 on M and N tilings.
static_assert((Tm % 2) == 0, "Tm must be even for 2x unroll");
static_assert((Tn % 2) == 0, "Tn must be even for 2x unroll");

void dense(
  XCAT(input_window_ ,DTYPE) * __restrict matA,
  XCAT(output_window_,DTYPE) * __restrict matC
){
  using MMUL = aie::mmul<mm_m, mm_k, mm_n, DTYPE, DTYPE>;

  const DTYPE* __restrict pA_base = (DTYPE*)matA->ptr;
  const DTYPE* __restrict pB_base = (DTYPE*)matB;
  DTYPE*       __restrict pC_base = (DTYPE*)matC->ptr;

#ifdef TILE_PROFILING
  aie::tile t = aie::tile::current();
  uint64 c0 = t.cycles();
#endif

  // Unroll by 2 in both M- and N-tile loops
  for (unsigned im = 0; im < Tm; im += 2) {

    const DTYPE * __restrict pA0_base = pA_base + ( (im    ) * Tk) * MMUL::size_A;
    const DTYPE * __restrict pA1_base = pA_base + ( (im + 1) * Tk) * MMUL::size_A;

    DTYPE * __restrict pC0 = pC_base + (im     * Tn) * MMUL::size_C;
    DTYPE * __restrict pC1 = pC_base + ((im+1) * Tn) * MMUL::size_C;

    for (unsigned in = 0; in < Tn; in += 2) {
      // A-pointers for the two M-tiles
      const DTYPE * __restrict pA0 = pA0_base;
      const DTYPE * __restrict pA1 = pA1_base;

      // B-pointers for the two N-tiles
      const DTYPE * __restrict pB0 = pB_base + ( 0 * Tn + in    ) * MMUL::size_B;
      const DTYPE * __restrict pB1 = pB_base + ( 0 * Tn + (in+1)) * MMUL::size_B;

      // Initial loads
      aie::vector<DTYPE, MMUL::size_A> A0 = aie::load_v<MMUL::size_A>(pA0); pA0 += MMUL::size_A;
      aie::vector<DTYPE, MMUL::size_A> A1 = aie::load_v<MMUL::size_A>(pA1); pA1 += MMUL::size_A;

      aie::vector<DTYPE, MMUL::size_B> B0 = aie::load_v<MMUL::size_B>(pB0); pB0 += MMUL::size_B * Tn;
      aie::vector<DTYPE, MMUL::size_B> B1 = aie::load_v<MMUL::size_B>(pB1); pB1 += MMUL::size_B * Tn;

      // Four accumulators (2x2 block)
      MMUL C00, C01, C10, C11;

      C00.mul(A0, B0); 
      C01.mul(A0, B1);
      C10.mul(A1, B0);
      C11.mul(A1, B1);

      // Inner-K accumulation
      for (unsigned ik = 0; ik < Tk - 1; ++ik) chess_flatten_loop
      {
        A0 = aie::load_v<MMUL::size_A>(pA0); pA0 += MMUL::size_A;
        A1 = aie::load_v<MMUL::size_A>(pA1); pA1 += MMUL::size_A;

        B0 = aie::load_v<MMUL::size_B>(pB0); pB0 += MMUL::size_B * Tn;
        B1 = aie::load_v<MMUL::size_B>(pB1); pB1 += MMUL::size_B * Tn;

        C00.mac(A0, B0);
        C01.mac(A0, B1);
        C10.mac(A1, B0);
        C11.mac(A1, B1);
      }

      auto v00 = C00.template to_vector<DTYPE>(SHIFT);
      auto v01 = C01.template to_vector<DTYPE>(SHIFT);
      auto v10 = C10.template to_vector<DTYPE>(SHIFT);
      auto v11 = C11.template to_vector<DTYPE>(SHIFT);

      auto o00 = DO_RELU ? aie::max(v00, (DTYPE)0) : v00;
      auto o01 = DO_RELU ? aie::max(v01, (DTYPE)0) : v01;
      auto o10 = DO_RELU ? aie::max(v10, (DTYPE)0) : v10;
      auto o11 = DO_RELU ? aie::max(v11, (DTYPE)0) : v11;

      aie::store_v(pC0, o00); pC0 += MMUL::size_C;
      aie::store_v(pC0, o01); pC0 += MMUL::size_C;
      aie::store_v(pC1, o10); pC1 += MMUL::size_C;
      aie::store_v(pC1, o11); pC1 += MMUL::size_C;
    }
  }

#ifdef TILE_PROFILING
  uint64 c1 = t.cycles();
  uint64 cycles = c1 - c0;
  uint64 macs = (uint64)(mm_M) * (uint64)(mm_K) * (uint64)(mm_N);
  uint64 cycles_expected = macs / 128;
  double efficiency = 100.0 * (double)cycles_expected / (double)cycles;

  printf("\n\n--------dense_i8 (2x2-unrolled) efficiency=(%.1f%%), cycles=%llu, cycles_expected=%llu "
         "(mm_m=%d mm_n=%d mm_k=%d Tm=%d Tk=%d Tn=%d SHIFT=%d)\n",
         efficiency, cycles, cycles_expected, mm_m, mm_n, mm_k, Tm, Tk, Tn, SHIFT);
#endif
}