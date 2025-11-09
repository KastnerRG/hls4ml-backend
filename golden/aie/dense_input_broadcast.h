#pragma once

#include <adf.h>
#include "aie_api/aie.hpp"

#ifndef LB_IN_M
#error "LB_IN_M must be defined before including dense_input_broadcast.h"
#endif
#ifndef LB_IN_K
#error "LB_IN_K must be defined before including dense_input_broadcast.h"
#endif
#ifndef LB_TILE_M
#error "LB_TILE_M must be defined before including dense_input_broadcast.h"
#endif
#ifndef LB_TILE_K
#error "LB_TILE_K must be defined before including dense_input_broadcast.h"
#endif
#ifndef O_TILES
#error "O_TILES must be defined before including dense_input_broadcast.h"
#endif

static_assert(O_TILES >= 1 && O_TILES <= 2, "dense_input_broadcast supports up to 2 outputs per kernel");
static_assert((LB_IN_M % LB_TILE_M) == 0, "LB_IN_M must be divisible by LB_TILE_M");
static_assert((LB_IN_K % LB_TILE_K) == 0, "LB_IN_K must be divisible by LB_TILE_K");

static constexpr int LB_IN_VEC = LB_TILE_M * LB_TILE_K;
static constexpr int LB_IN_TOTAL_VEC = (LB_IN_M * LB_IN_K) / LB_IN_VEC;

static inline void dense_input_broadcast(
    input_stream_int8 * __restrict in,
    output_stream_int8 * __restrict * __restrict outs)
{
  for (int vec = 0; vec < LB_IN_TOTAL_VEC; ++vec)
  chess_prepare_for_pipelining
  {
    aie::vector<int8, LB_IN_VEC> v = readincr_v<LB_IN_VEC>(in);
    for (int t = 0; t < O_TILES; ++t)
    {
      writeincr(outs[t], v);
    }
  }
}
