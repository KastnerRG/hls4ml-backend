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
using ACC = typename MM::accum_type;

#ifndef NB
#define NB 2
#endif
static_assert(NB <= 2, "dense_stream_out currently supports NB up to 2");

#ifndef DENSE_IN_CASC_TYPE
#define DENSE_IN_CASC_TYPE acc32
#endif

using input_stream_t = STREAM_CAT(input_stream_, DTYPE);
using output_port_t = STREAM_CAT(output_stream_, DTYPE);
using input_cascade_t = input_cascade<DENSE_IN_CASC_TYPE>;
using output_cascade_t = output_cascade<DENSE_IN_CASC_TYPE>;
using cascade_vec_t = aie::accum<DENSE_IN_CASC_TYPE, MM::size_A>;

template<typename CascTag, unsigned Lanes>
static inline aie::accum<CascTag, Lanes> read_cascade_helper(input_cascade<CascTag> *__restrict c) {
  return aie::detail::adf::cascade_stream_helper<CascTag, Lanes>::readincr(c);
}

static inline VA cascade_to_vector(const cascade_vec_t &acc) {
  return acc.template to_vector<DTYPE>();
}

static inline cascade_vec_t vector_to_cascade(const VA &v) {
  cascade_vec_t acc;
  acc.from_vector(v);
  return acc;
}

template<unsigned Slots>
static inline void process_block_out(const VA *Abuf,
                                     const DTYPE* __restrict Bbase,
                                     unsigned base_offset,
                                     unsigned strideB_perK,
                                     output_port_t * __restrict sC)
{
  if constexpr (Slots == 0) {
    return;
  }

  MM C0;
  MM C1;

  {
    const VA A0 = Abuf[0];
    const DTYPE* __restrict pB0 = Bbase + base_offset;

    VB b0 = aie::load_v<MM::size_B>(pB0); pB0 += MM::size_B;
    C0.mul(A0, b0);
    if constexpr (Slots >= 2) {
      VB b1 = aie::load_v<MM::size_B>(pB0); pB0 += MM::size_B;
      C1.mul(A0, b1);
    }
  }

  for (unsigned ik = 1; ik < Tk; ++ik)
  chess_prepare_for_pipelining chess_loop_range(1,)
  {
    const VA A  = Abuf[ik];
    const DTYPE* __restrict pBk = Bbase + (ik * strideB_perK) + base_offset;

    VB b0 = aie::load_v<MM::size_B>(pBk); pBk += MM::size_B;
    C0.mac(A, b0);
    if constexpr (Slots >= 2) {
      VB b1 = aie::load_v<MM::size_B>(pBk); pBk += MM::size_B;
      C1.mac(A, b1);
    }
  }

  {
    ACC acc0 = C0;
    VC v = aie::to_vector<DTYPE>(acc0, SHIFT);
    if (DO_RELU) v = aie::max(v,(DTYPE)0);
    writeincr(sC, v);
  }

  if constexpr (Slots >= 2) {
    ACC acc1 = C1;
    VC v = aie::to_vector<DTYPE>(acc1, SHIFT);
    if (DO_RELU) v = aie::max(v,(DTYPE)0);
    writeincr(sC, v);
  }
}

enum class DenseOutMode { First, Middle, Last, Single };

template<DenseOutMode Mode>
static inline void dense_out_kernel(input_stream_t * __restrict sA,
                                    input_cascade_t * __restrict casc_in,
                                    output_cascade_t * __restrict casc_out,
                                    output_port_t * __restrict sC)
{
  const DTYPE* __restrict Bbase = (const DTYPE*)matB;
  constexpr unsigned blocksN = (Tn + NB - 1) / NB;
  const unsigned stride_block = NB * MM::size_B;
  const unsigned strideB_perK = blocksN * stride_block;
  constexpr unsigned full_blocks = Tn / NB;
  constexpr unsigned tail_slots  = Tn % NB;
  constexpr bool HasStreamIn   = (Mode == DenseOutMode::First) || (Mode == DenseOutMode::Single);
  constexpr bool HasCascadeIn  = (Mode == DenseOutMode::Middle) || (Mode == DenseOutMode::Last);
  constexpr bool HasCascadeOut = (Mode == DenseOutMode::First)  || (Mode == DenseOutMode::Middle);

#ifdef FREE
  while(1) {
#endif

  for (unsigned im = 0; im < Tm; ++im)
  chess_prepare_for_pipelining chess_loop_range(1,)
  {
    VA Abuf[Tk];

    if constexpr (HasStreamIn) {
      for (unsigned ik = 0; ik < Tk; ++ik)
      chess_prepare_for_pipelining chess_loop_range(1,)
      {
        Abuf[ik] = readincr_v<MM::size_A>(sA);
        if constexpr (HasCascadeOut) {
          auto casc_val = vector_to_cascade(Abuf[ik]);
          writeincr(casc_out, casc_val);
        }
      }
    } else {
      for (unsigned ik = 0; ik < Tk; ++ik)
      chess_prepare_for_pipelining chess_loop_range(1,)
      {
        auto casc_val = read_cascade_helper<DENSE_IN_CASC_TYPE, MM::size_A>(casc_in);
        Abuf[ik] = cascade_to_vector(casc_val);
        if constexpr (HasCascadeOut) {
          writeincr(casc_out, casc_val);
        }
      }
    }

    for (unsigned blk = 0; blk < full_blocks; ++blk)
    chess_prepare_for_pipelining chess_loop_range(1,)
    {
      const unsigned base_offset = blk * stride_block;
      process_block_out<NB>(Abuf, Bbase, base_offset, strideB_perK, sC);
    }

    if constexpr (tail_slots != 0) {
      const unsigned base_offset = full_blocks * stride_block;
      process_block_out<tail_slots>(Abuf, Bbase, base_offset, strideB_perK, sC);
    }
  }

#ifdef FREE
  }
#endif
}

static inline void dense_out_first(input_stream_t * __restrict sA,
                                   output_cascade_t * __restrict casc_out,
                                   output_port_t * __restrict sC) {
  dense_out_kernel<DenseOutMode::First>(sA, nullptr, casc_out, sC);
}

static inline void dense_out_middle(input_cascade_t * __restrict casc_in,
                                    output_cascade_t * __restrict casc_out,
                                    output_port_t * __restrict sC) {
  dense_out_kernel<DenseOutMode::Middle>(nullptr, casc_in, casc_out, sC);
}

static inline void dense_out_last(input_cascade_t * __restrict casc_in,
                                  output_port_t * __restrict sC) {
  dense_out_kernel<DenseOutMode::Last>(nullptr, casc_in, nullptr, sC);
}

static inline void dense_out_single(input_stream_t * __restrict sA,
                                    output_port_t * __restrict sC) {
  dense_out_kernel<DenseOutMode::Single>(sA, nullptr, nullptr, sC);
}

#undef DENSE_IN_CASC_TYPE
