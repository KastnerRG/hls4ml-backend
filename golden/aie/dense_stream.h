#include <adf.h>
#include "aie_api/aie.hpp"
#include "aie_api/aie_adf.hpp"

#define CAT(a,b) a##b
#define XCAT(a,b) CAT(a,b)
#define STREAM_CAT(prefix, type) CAT(prefix, type)

#define Tm (mm_M / mm_m)
#define Tk (mm_K / mm_k)
#define Tn (mm_N / mm_n)

static_assert(mm_M % mm_m == 0 && mm_K % mm_k == 0 && mm_N % mm_n == 0, "Bad tiling");

using MM = aie::mmul<mm_m, mm_k, mm_n, DTYPE, DTYPE>;
using VA = aie::vector<DTYPE, MM::size_A>;
using VB = aie::vector<DTYPE, MM::size_B>;
using VC = aie::vector<DTYPE, MM::size_C>;
using VI = aie::vector<int32, MM::size_C>;

// Tune at build time with -DNB=2/4/8 as needed
#ifndef NB
#define NB 4
#endif
using input_stream_t = STREAM_CAT(input_stream_, DTYPE);
using output_port_t = STREAM_CAT(output_stream_, DTYPE);
using ACC = typename MM::accum_type;
using ACC_TAG = typename ACC::value_type;
#ifndef DENSE_CASC_TYPE
#define DENSE_CASC_TYPE ACC_TAG
#endif
using input_cascade_t = input_cascade<DENSE_CASC_TYPE>;
using output_cascade_t = output_cascade<DENSE_CASC_TYPE>;

template<typename CascTag, unsigned Lanes>
static inline aie::accum<CascTag, Lanes> read_cascade(input_cascade<CascTag> *__restrict c) {
  return aie::detail::adf::cascade_stream_helper<CascTag, Lanes>::readincr(c);
}

enum class DenseMode { First, Middle, Last, Single };

template<DenseMode Mode>
static inline void dense_kernel(input_stream_t * __restrict sA,
                                input_cascade_t * __restrict casc_in,
                                output_cascade_t * __restrict casc_out,
                                output_port_t * __restrict sC)
{
  constexpr bool HasCascadeIn   = (Mode == DenseMode::Middle) || (Mode == DenseMode::Last);
  constexpr bool HasCascadeOut  = (Mode == DenseMode::First)  || (Mode == DenseMode::Middle);
  constexpr bool HasStreamOut   = (Mode == DenseMode::Last)   || (Mode == DenseMode::Single);
  const DTYPE* __restrict Bbase = (const DTYPE*)matB;
  const unsigned strideB_perK  = MM::size_B * Tn;   // bytes to jump between successive K-slices
  const unsigned blocksN       = (Tn + NB - 1) / NB;

#ifdef FREE
  while(1) {
#endif

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
        const DTYPE* __restrict pB0 = Bbase + (0 * strideB_perK) + in0 * MM::size_B;

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
        const DTYPE* __restrict pBk = Bbase + (ik * strideB_perK) + in0 * MM::size_B;

        if (n_this >= 1) { VB b0 = aie::load_v<MM::size_B>(pBk); pBk += MM::size_B; C0.mac(A, b0); }
        if (n_this >= 2) { VB b1 = aie::load_v<MM::size_B>(pBk); pBk += MM::size_B; C1.mac(A, b1); }
        if (n_this >= 3) { VB b2 = aie::load_v<MM::size_B>(pBk); pBk += MM::size_B; C2.mac(A, b2); }
        if (n_this >= 4) { VB b3 = aie::load_v<MM::size_B>(pBk);                    C3.mac(A, b3); }
      }

      // ---- Quantize/cascade (+ReLU) immediately for this block
      if (n_this >= 1) {
        ACC acc0 = C0;
        if constexpr (HasCascadeIn) {
          auto casc_val = read_cascade<DENSE_CASC_TYPE, MM::size_C>(casc_in);
          acc0 = aie::add(acc0, casc_val);
        }
        if constexpr (HasCascadeOut) {
          writeincr(casc_out, acc0);
        } else if constexpr (HasStreamOut) {
          VC v = aie::to_vector<DTYPE>(acc0, SHIFT);
          if (DO_RELU) v = aie::max(v,(DTYPE)0);
          writeincr(sC, v);
        }
      }
      if (n_this >= 2) {
        ACC acc1 = C1;
        if constexpr (HasCascadeIn) {
          auto casc_val = read_cascade<DENSE_CASC_TYPE, MM::size_C>(casc_in);
          acc1 = aie::add(acc1, casc_val);
        }
        if constexpr (HasCascadeOut) {
          writeincr(casc_out, acc1);
        } else if constexpr (HasStreamOut) {
          VC v = aie::to_vector<DTYPE>(acc1, SHIFT);
          if (DO_RELU) v = aie::max(v,(DTYPE)0);
          writeincr(sC, v);
        }
      }
      if (n_this >= 3) {
        ACC acc2 = C2;
        if constexpr (HasCascadeIn) {
          auto casc_val = read_cascade<DENSE_CASC_TYPE, MM::size_C>(casc_in);
          acc2 = aie::add(acc2, casc_val);
        }
        if constexpr (HasCascadeOut) {
          writeincr(casc_out, acc2);
        } else if constexpr (HasStreamOut) {
          VC v = aie::to_vector<DTYPE>(acc2, SHIFT);
          if (DO_RELU) v = aie::max(v,(DTYPE)0);
          writeincr(sC, v);
        }
      }
      if (n_this >= 4) {
        ACC acc3 = C3;
        if constexpr (HasCascadeIn) {
          auto casc_val = read_cascade<DENSE_CASC_TYPE, MM::size_C>(casc_in);
          acc3 = aie::add(acc3, casc_val);
        }
        if constexpr (HasCascadeOut) {
          writeincr(casc_out, acc3);
        } else if constexpr (HasStreamOut) {
          VC v = aie::to_vector<DTYPE>(acc3, SHIFT);
          if (DO_RELU) v = aie::max(v,(DTYPE)0);
          writeincr(sC, v);
        }
      }
    }
  }

#ifdef TILE_PROFILING
  uint64 c1 = t.cycles();
  uint64 cycles = c1 - c0;
  uint64 macs = (uint64)mm_M * (uint64)mm_K * (uint64)mm_N;
  uint64 exp  = macs / 128;
  printf("\n[dense_stream NB=%d] eff=%.1f%% cycles=%llu exp=%llu "
         "(m=%d n=%d k=%d  Tm=%d Tk=%d Tn=%d SHIFT=%d)\n",
         (int)NB, 100.0*(double)exp/(double)cycles, cycles, exp,
         mm_m, mm_n, mm_k, Tm, Tk, Tn, SHIFT);
#endif

#ifdef FREE
  }
#endif
}

static inline void dense_first(input_stream_t * __restrict sA,
                               output_cascade_t * __restrict casc_out) {
  dense_kernel<DenseMode::First>(sA, nullptr, casc_out, nullptr);
}

static inline void dense_middle(input_stream_t * __restrict sA,
                                input_cascade_t * __restrict casc_in,
                                output_cascade_t * __restrict casc_out) {
  dense_kernel<DenseMode::Middle>(sA, casc_in, casc_out, nullptr);
}

static inline void dense_last(input_stream_t * __restrict sA,
                              input_cascade_t * __restrict casc_in,
                              output_port_t * __restrict sC) {
  dense_kernel<DenseMode::Last>(sA, casc_in, nullptr, sC);
}

static inline void dense_single(input_stream_t * __restrict sA,
                                output_port_t * __restrict sC) {
  dense_kernel<DenseMode::Single>(sA, nullptr, nullptr, sC);
}

#undef DENSE_CASC_TYPE
