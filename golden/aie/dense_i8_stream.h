// dense_i8_stream.h — streaming GEMM, NB=4 blocked across N (low-latency, no stalls)
#include <adf.h>
#include "aie_api/aie.hpp"
#include "aie_api/aie_adf.hpp"

#define Tm (mm_M / mm_m)
#define Tk (mm_K / mm_k)
#define Tn (mm_N / mm_n)

static_assert(mm_M % mm_m == 0 && mm_K % mm_k == 0 && mm_N % mm_n == 0, "Bad tiling");

using MM = aie::mmul<mm_m, mm_k, mm_n, int8, int8>;
using VA = aie::vector<int8, MM::size_A>;
using VB = aie::vector<int8, MM::size_B>;
using VC = aie::vector<int8, MM::size_C>;

// Tune at build time with -DNB=2/4/8 as needed
#ifndef NB
#define NB 4
#endif

static inline void dense_i8(input_stream_int8  * __restrict sA,
                            output_stream_int8 * __restrict sC)
{
  const int8* __restrict Bbase = (const int8*)matB;
  const unsigned strideB_perK  = MM::size_B * Tn;   // bytes to jump between successive K-slices
  const unsigned blocksN       = (Tn + NB - 1) / NB;

#ifdef TILE_PROFILING
  aie::tile t = aie::tile::current();
  uint64 c0 = t.cycles();
#endif

  // Iterate M-tiles
  for (unsigned im = 0; im < Tm; ++im)
  chess_prepare_for_pipelining chess_loop_range(1,)
  {
    // Buffer A for this M row once: Tk * size_A
    VA Abuf[Tk];
    for (unsigned ik = 0; ik < Tk; ++ik)
    chess_prepare_for_pipelining chess_loop_range(1,)
    {
      Abuf[ik] = readincr_v<MM::size_A>(sA);
    }

    // Walk N in NB-sized blocks; emit each block as soon as its K loop is done
    for (unsigned blk = 0; blk < blocksN; ++blk)
    chess_prepare_for_pipelining chess_loop_range(1,)
    {
      const unsigned in0    = blk * NB;
      const unsigned n_this = (in0 + NB <= Tn) ? NB : (Tn - in0);

      // Up to NB accumulators (specialized with if-guards)
      MM C0, C1, C2, C3;

      // ---- K = 0: initialize accumulators with MUL
      {
        const VA A0 = Abuf[0];
        const int8* __restrict pB0 = Bbase + (0 * strideB_perK) + in0 * MM::size_B;

        if (n_this >= 1) { VB b0 = aie::load_v<MM::size_B>(pB0); pB0 += MM::size_B; C0.mul(A0, b0); }
        if (n_this >= 2) { VB b1 = aie::load_v<MM::size_B>(pB0); pB0 += MM::size_B; C1.mul(A0, b1); }
        if (n_this >= 3) { VB b2 = aie::load_v<MM::size_B>(pB0); pB0 += MM::size_B; C2.mul(A0, b2); }
        if (n_this >= 4) { VB b3 = aie::load_v<MM::size_B>(pB0);                    C3.mul(A0, b3); }
      }

      // ---- K = 1..Tk-1: MAC
      for (unsigned ik = 1; ik < Tk; ++ik)
      chess_prepare_for_pipelining chess_loop_range(1,)
      {
        const VA A  = Abuf[ik];
        const int8* __restrict pBk = Bbase + (ik * strideB_perK) + in0 * MM::size_B;

        if (n_this >= 1) { VB b0 = aie::load_v<MM::size_B>(pBk); pBk += MM::size_B; C0.mac(A, b0); }
        if (n_this >= 2) { VB b1 = aie::load_v<MM::size_B>(pBk); pBk += MM::size_B; C1.mac(A, b1); }
        if (n_this >= 3) { VB b2 = aie::load_v<MM::size_B>(pBk); pBk += MM::size_B; C2.mac(A, b2); }
        if (n_this >= 4) { VB b3 = aie::load_v<MM::size_B>(pBk);                    C3.mac(A, b3); }
      }

      // ---- Quantize (+ReLU) and stream out immediately for this block
      if (n_this >= 1) { VC v = C0.template to_vector<int8>(SHIFT); if (DO_RELU) v = aie::max(v,(int8)0); writeincr(sC, v); }
      if (n_this >= 2) { VC v = C1.template to_vector<int8>(SHIFT); if (DO_RELU) v = aie::max(v,(int8)0); writeincr(sC, v); }
      if (n_this >= 3) { VC v = C2.template to_vector<int8>(SHIFT); if (DO_RELU) v = aie::max(v,(int8)0); writeincr(sC, v); }
      if (n_this >= 4) { VC v = C3.template to_vector<int8>(SHIFT); if (DO_RELU) v = aie::max(v,(int8)0); writeincr(sC, v); }
    }
  }

#ifdef TILE_PROFILING
  uint64 c1 = t.cycles();
  uint64 cycles = c1 - c0;
  uint64 macs = (uint64)mm_M * (uint64)mm_K * (uint64)mm_N;
  uint64 exp  = macs / 128;
  printf("\n[dense_i8_stream NB=%d] eff=%.1f%% cycles=%llu exp=%llu "
         "(m=%d n=%d k=%d  Tm=%d Tk=%d Tn=%d SHIFT=%d)\n",
         (int)NB, 100.0*(double)exp/(double)cycles, cycles, exp,
         mm_m, mm_n, mm_k, Tm, Tk, Tn, SHIFT);
#endif
}
