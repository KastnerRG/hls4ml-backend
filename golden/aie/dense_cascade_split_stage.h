#pragma once

#include <adf.h>
#include "aie_api/aie.hpp"
#include "aie_api/aie_adf.hpp"

#ifndef DENSE_CASC_TYPE
#error "DENSE_CASC_TYPE must be defined before including dense_cascade_split_stage.h"
#endif
#ifndef SPLIT_TK
#error "SPLIT_TK must be defined before including dense_cascade_split_stage.h"
#endif
#ifndef TK_RESIDUAL
#error "TK_RESIDUAL must be defined before including dense_cascade_split_stage.h"
#endif
#ifndef HAS_CASC_OUT
#error "HAS_CASC_OUT must be defined before including dense_cascade_split_stage.h"
#endif

using input_cascade_t = input_cascade<DENSE_CASC_TYPE>;
using output_cascade_t = output_cascade<DENSE_CASC_TYPE>;
#define CAT(a,b) a##b
#define STREAM_CAT(prefix, type) CAT(prefix, type)
using output_stream_t = STREAM_CAT(output_stream_, DTYPE);

using MM = aie::mmul<mm_m, mm_k, mm_n, DTYPE, DTYPE>;
using VC = aie::vector<DTYPE, MM::size_C>;

static_assert(mm_M % mm_m == 0 && mm_K % mm_k == 0, "Bad tiling");

constexpr unsigned Tm = mm_M / mm_m;
constexpr unsigned Tk = TK_RESIDUAL;

template<typename CascTag, unsigned Lanes>
static inline aie::accum<CascTag, Lanes> read_cascade_helper(input_cascade<CascTag> *__restrict c) {
  return aie::detail::adf::cascade_stream_helper<CascTag, Lanes>::readincr(c);
}

template<typename CascTag, unsigned Lanes>
static inline void write_cascade_helper(output_cascade<CascTag> *__restrict c,
                                        const aie::accum<CascTag, Lanes> &acc) {
  aie::detail::adf::cascade_stream_helper<CascTag, Lanes>::writeincr(c, acc);
}

static inline void dense_cascade_split_stage(input_cascade_t * __restrict casc_in,
#if HAS_CASC_OUT
                                             output_cascade_t * __restrict casc_out,
#endif
                                             output_stream_t * __restrict out)
{
  for (unsigned im = 0; im < Tm; ++im)
  chess_prepare_for_pipelining chess_loop_range(1,)
  {
    for (unsigned ik = 0; ik < Tk; ++ik)
    chess_prepare_for_pipelining chess_loop_range(1,)
    {
      auto acc = read_cascade_helper<DENSE_CASC_TYPE, MM::size_C>(casc_in);
      if (ik < SPLIT_TK) {
        auto v = aie::to_vector<DTYPE>(acc, SHIFT);
        if (DO_RELU) v = aie::max(v, (DTYPE)0);
        writeincr(out, v);
      }
#if HAS_CASC_OUT
      else {
        write_cascade_helper<DENSE_CASC_TYPE, MM::size_C>(casc_out, acc);
      }
#endif
    }
  }
}

#undef STREAM_CAT
#undef CAT
