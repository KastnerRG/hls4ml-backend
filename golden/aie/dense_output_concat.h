#pragma once

#include <adf.h>
#include "aie_api/aie.hpp"

#ifndef LB_OUT_M
#error "LB_OUT_M must be defined before including dense_output_concat.h"
#endif
#ifndef LB_OUT_N
#error "LB_OUT_N must be defined before including dense_output_concat.h"
#endif
#ifndef LB_TILE_M
#error "LB_TILE_M must be defined before including dense_output_concat.h"
#endif
#ifndef LB_TILE_N
#error "LB_TILE_N must be defined before including dense_output_concat.h"
#endif
#ifndef O_TILES
#error "O_TILES must be defined before including dense_output_concat.h"
#endif

static_assert(O_TILES >= 1 && O_TILES <= 2, "dense_output_concat supports up to 2 inputs per kernel");
static_assert((LB_OUT_M % LB_TILE_M) == 0, "LB_OUT_M must be divisible by LB_TILE_M");
static_assert((LB_OUT_N % O_TILES) == 0, "LB_OUT_N must be divisible by O_TILES");

static constexpr int LB_OUT_TILE_VEC = LB_TILE_M * LB_TILE_N;
static_assert((LB_OUT_N / O_TILES) % LB_TILE_N == 0, "Tile columns must align with LB_TILE_N");

static constexpr int LB_TM = LB_OUT_M / LB_TILE_M;               // number of row tiles
static constexpr int LB_VEC_PER_ROW = (LB_OUT_N / O_TILES) / LB_TILE_N;
static constexpr int LB_OUT_VECS_PER_TILE = LB_TM * LB_VEC_PER_ROW;

static inline void dense_output_concat(
    input_stream_int8 * __restrict * __restrict ins,
    output_stream_int8 * __restrict out)
{
  for (int tm = 0; tm < LB_TM; ++tm)
  {
    for (int tile = 0; tile < O_TILES; ++tile)
    {
      for (int vec = 0; vec < LB_VEC_PER_ROW; ++vec)
      chess_prepare_for_pipelining
      {
        aie::vector<int8, LB_OUT_TILE_VEC> v = readincr_v<LB_OUT_TILE_VEC>(ins[tile]);
        writeincr(out, v);
      }
    }
  }
}
