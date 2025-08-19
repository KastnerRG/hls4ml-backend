// dense_i8_stream.h
#include <adf.h>
#include "aie_api/aie.hpp"
#include "aie_api/aie_adf.hpp"

// mm_* and DO_RELU/SHIFT are provided by the layer_*.cc before including this header.
#define Tm (mm_M / mm_m)
#define Tk (mm_K / mm_k)
#define Tn (mm_N / mm_n)

static_assert(mm_M % mm_m == 0 && mm_K % mm_k == 0 && mm_N % mm_n == 0, "Bad tiling");
using i8vA = aie::vector<int8, aie::mmul<mm_m,mm_k,mm_n,int8,int8>::size_A>;
using i8vB = aie::vector<int8, aie::mmul<mm_m,mm_k,mm_n,int8,int8>::size_B>;
using i8vC = aie::vector<int8, aie::mmul<mm_m,mm_k,mm_n,int8,int8>::size_C>;

static inline void dense_i8(
  input_stream_int8  * __restrict sA,
  output_stream_int8 * __restrict sC)
{
  using MM = aie::mmul<mm_m,mm_k,mm_n,int8,int8>;

  // B is pre-tiled as [ik-major][in-minor] blocks of size_B int8
  const int8 * __restrict pB_base = (const int8*)matB;

#ifdef TILE_PROFILING
  aie::tile t = aie::tile::current();
  uint64 c0 = t.cycles();
#endif

  // One invocation of this kernel consumes exactly Tm*Tk A-tiles
  // and produces exactly Tm*Tn C-tiles.
  for (unsigned im = 0; im < Tm; ++im)
  {
    // Read all K-tiles for this M-tile from the input stream once.
    i8vA Atiles[Tk];
    for (unsigned ik = 0; ik < Tk; ++ik) {
      Atiles[ik] = readincr_v<MM::size_A>(sA);
    }

    // For each N-tile, reuse the cached Atiles[ik] across the K-loop.
    for (unsigned in = 0; in < Tn; ++in)
    {
      MM C;

      // First K-tile: mul
      {
        const int8 *pB0 = pB_base + (0 * Tn + in) * MM::size_B;
        i8vB B0 = aie::load_v<MM::size_B>(pB0);
        C.mul(Atiles[0], B0);
      }

      // Remaining K-tiles: mac
      for (unsigned ik = 1; ik < Tk; ++ik) chess_flatten_loop {
        const int8 *pBk = pB_base + (ik * Tn + in) * MM::size_B;
        i8vB Bk = aie::load_v<MM::size_B>(pBk);
        C.mac(Atiles[ik], Bk);
      }

      // Quantize (SHIFT), optional ReLU, and stream out
      i8vC v = C.template to_vector<int8>(SHIFT);
      if (DO_RELU) v = aie::max(v, (int8)0);
      writeincr_v(sC, v);
    }
  }

#ifdef TILE_PROFILING
  uint64 c1 = t.cycles();
  uint64 cycles = c1 - c0;
  uint64 macs = (uint64)mm_M * (uint64)mm_K * (uint64)mm_N;
  uint64 cycles_expected = macs / 128; // 128 int8 MACs/cycle for mmul
  double eff = 100.0 * (double)cycles_expected / (double)cycles;

  printf("\n[dense_i8_stream] eff=%.1f%%  cycles=%llu  exp=%llu  "
         "(m=%d n=%d k=%d  Tm=%d Tk=%d Tn=%d SHIFT=%d)\n",
         eff, cycles, cycles_expected, mm_m, mm_n, mm_k, Tm, Tk, Tn, SHIFT);
#endif
}
