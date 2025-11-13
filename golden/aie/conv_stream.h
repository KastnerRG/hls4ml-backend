#include <adf.h>
#include <cstdint>
#include "aie_api/aie.hpp"
#include "aie_api/aie_adf.hpp"

using MM   = aie::mmul<2, 8, 8, int8, int8>;
using VA16 = aie::vector<int8, 16>;
using VB64 = aie::vector<int8, 64>;
using VC16 = aie::vector<int8, 16>;

#ifndef DO_RELU
#define DO_RELU false
#endif

extern const int8_t k_p[];

constexpr int XC8 = (CI + 7) / 8;
constexpr int YC8 = CO / 8;
static_assert(CO % 8 == 0, "Output channels must be a multiple of 8");
static_assert(KH > 0 && KW > 0, "Kernel dimensions must be positive");
static_assert(SH > 0 && SW > 0, "Strides must be positive");
static_assert((YH % 2) == 0, "YH must be even for this streaming kernel (m=2)");

constexpr int TAP_BYTES    = 64; // 8 (ci lanes) * 8 (co lanes)
constexpr int B_stride_yc8 = KH * KW * TAP_BYTES;
constexpr int B_stride_xc8 = YC8 * B_stride_yc8;
constexpr int RB_ROWS      = KH + SH;

static inline void stream_row(
    output_stream_int8* __restrict s_out,
    const int8* __restrict row_ptr,
    int total_bytes)
{
  static_assert((YW * CO) % 16 == 0, "Output row bytes must be multiple of 16");
  for (int idx = 0; idx < total_bytes; idx += 16)
  chess_prepare_for_pipelining chess_loop_range(1,)
  {
    auto v = aie::load_v<16>(row_ptr + idx);
    writeincr(s_out, v);
  }
}

static inline VA16 pack_tb16_from_rb(
    int abs_top,
    int xw,
    int xc8,
    int  row_tags[RB_ROWS],
    int8 (&rb)[RB_ROWS][XW * CI])
{
  alignas(32) int8 tmp[16];

  auto fetch_lane = [&](int abs_row, int lane) -> int8 {
    if (abs_row < 0 || abs_row >= XH) return 0;
    if (xw < 0 || xw >= XW) return 0;
    const int xc = xc8 * 8 + lane;
    if (xc >= CI) return 0;
    const int slot = abs_row % RB_ROWS;
    if (row_tags[slot] != abs_row) return 0;
    const int idx = (xw * CI) + xc;
    return rb[slot][idx];
  };

  for (int lane = 0; lane < 8; ++lane)
  chess_prepare_for_pipelining chess_loop_range(1,)
  {
    tmp[lane] = fetch_lane(abs_top, lane);
  }

  const int abs_bot = abs_top + SH;
  for (int lane = 0; lane < 8; ++lane)
  chess_prepare_for_pipelining chess_loop_range(1,)
  {
    tmp[8 + lane] = fetch_lane(abs_bot, lane);
  }

  return aie::load_v<16>(tmp);
}

static inline void load_rows_until(
    input_stream_int8* __restrict s_in,
    int need_max,
    int &rows_ready,
    int8 (&rb)[RB_ROWS][XW * CI],
    int  row_tags[RB_ROWS])
{
  const int row_bytes = XW * CI;
  static_assert((XW * CI) % 16 == 0, "Row bytes must be multiple of 16");

  while (rows_ready < need_max && (rows_ready + 1) < XH)
  chess_prepare_for_pipelining chess_loop_range(1,)
  {
    const int abs_h = rows_ready + 1;
    const int slot  = abs_h % RB_ROWS;
    int8* __restrict dst = &rb[slot][0];

    for (int copied = 0; copied < row_bytes; copied += 16)
    chess_prepare_for_pipelining chess_loop_range(1,)
    {
      auto v = readincr_v<16>(s_in);
      aie::store_v(&dst[copied], v);
    }

    row_tags[slot] = abs_h;
    rows_ready = abs_h;
  }
}

static inline void conv_stream(
    input_stream_int8* __restrict s_in,
    output_stream_int8* __restrict s_out)
{
  alignas(32) static int8 rb[RB_ROWS][XW * CI];
  alignas(32) static int8 row_buf[2][YW * CO];

#ifdef FREE
  while (1) {
#endif

    int row_tags[RB_ROWS];
    for (int i = 0; i < RB_ROWS; ++i)
    chess_prepare_for_pipelining chess_loop_range(1,)
    { row_tags[i] = -1; }

    int rows_ready = -1;

    for (int yh2 = 0; yh2 < YH; yh2 += 2)
    chess_prepare_for_pipelining chess_loop_range(1,)
    {
      const int need_top_max = yh2 * SH - PH + (KH - 1);
      const int need_bot_max = need_top_max + SH;
      const int need_max     = need_bot_max;

      load_rows_until(s_in, need_max, rows_ready, rb, row_tags);

      for (int yw = 0; yw < YW; ++yw)
      chess_prepare_for_pipelining chess_loop_range(1,)
      {
        for (int yc8 = 0; yc8 < YC8; ++yc8)
        chess_prepare_for_pipelining chess_loop_range(1,)
        {
          MM C;
          bool first = true;

          for (int xc8 = 0; xc8 < XC8; ++xc8)
          chess_prepare_for_pipelining chess_loop_range(1,)
          {
            const int8_t* __restrict pBci = k_p + xc8 * B_stride_xc8 + yc8 * B_stride_yc8;

            for (int kh = 0; kh < KH; ++kh)
            chess_prepare_for_pipelining chess_loop_range(1,)
            {
              const int xh_top = yh2 * SH - PH + kh;

              for (int kw = 0; kw < KW; ++kw)
              chess_prepare_for_pipelining chess_loop_range(1,)
              {
                const int xw = yw * SW - PW + kw;

                VA16 Av = pack_tb16_from_rb(xh_top, xw, xc8, row_tags, rb);
                VB64 Bv = aie::load_v<64>(pBci);  pBci += TAP_BYTES;

                if (first) { C.mul(Av, Bv); first = false; }
                else       { C.mac(Av, Bv); }
              }
            }
          }

          auto vec32 = C.template to_vector<int32_t>(0);
          alignas(32) int32_t acc32[16];
          aie::store_v(acc32, vec32);

          alignas(32) int8 bytes16[16];
          for (int i = 0; i < 16; ++i)
          chess_prepare_for_pipelining chess_loop_range(1,)
          {
            int32_t v = acc32[i];
#if SHIFT > 0
            v = (v + (1 << (SHIFT - 1))) >> SHIFT;
#endif
            int8_t q = static_cast<int8_t>(v);
            if (DO_RELU && q < 0) q = 0;
            bytes16[i] = q;
          }

          const int offset = yw * CO + yc8 * 8;
          __builtin_memcpy(&row_buf[0][offset], &bytes16[0], 8);
          __builtin_memcpy(&row_buf[1][offset], &bytes16[8], 8);
        }
      }

      for (int r = 0; r < 2; ++r)
      chess_prepare_for_pipelining chess_loop_range(1,)
      {
        if ((yh2 + r) >= YH) break;
        stream_row(s_out, &row_buf[r][0], YW * CO);
      }
    }

#ifdef FREE
  }
#endif
}
