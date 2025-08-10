#include <adf.h>
#include "aie_api/aie.hpp"
#include "aie_api/aie_adf.hpp"

void dense_i8(
  input_window_int8 * __restrict matA,
  output_window_int8 * __restrict matC
){
  using MMUL = aie::mmul<m_api, k_api, n_api, int8, int8>;

  const int8* __restrict pA = (int8*)matA->ptr;
  const int8* __restrict pB = (int8*)matB;
  int8*       __restrict pC = (int8*)matC->ptr;

  aie::tile t = aie::tile::current();
  uint64 c0 = t.cycles();

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
      auto C_out = DO_RELU ? aie::max(C_vec, (int8)0) : C_vec;
      aie::store_v(pC, C_out); pC += MMUL::size_C;
    }
  }

  uint64 c1 = t.cycles();
  uint64 cycles = c1 - c0;
  uint64 macs = (uint64)(m_api*Tm) * (uint64)(k_api*Tk) * (uint64)(n_api*Tn);
  uint64 cycles_expected = macs / 128;
  double efficiency = 100* (double)cycles_expected / cycles;
  printf("\n\n-----------dense_i8 efficiency=(%.1f%%), cycles=%llu, cycles_expected=%llu (m_api=%d n_api=%d k_api=%d Tm=%d Tk=%d Tn=%d SHIFT=%d)\n",
         efficiency, cycles, cycles_expected, m_api, n_api, k_api, Tm, Tk, Tn, SHIFT);
}