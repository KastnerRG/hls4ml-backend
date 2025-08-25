// int32→int32 streaming Conv2D (m=2,k=8,n=8), supports row-major or TB16 input
#include <adf.h>
#include "aie_api/aie.hpp"
#include "aie_api/aie_adf.hpp"

#ifndef IN_TB16
#define IN_TB16 0   // 0: row-major int32 (PLIO), 1: TB16 int32 from previous Conv
#endif

static_assert((XC % 8) == 0 && (YC % 8) == 0, "XC & YC must be multiples of 8");
static_assert(SH >= 1 && SW >= 1,             "Strides must be >= 1");
static_assert(((XW*XC) % 4) == 0,              "Row bytes must be multiple of 4");

using MM   = aie::mmul<2,8,8,int8,int8>;
using VA16 = aie::vector<int8,16>;
using VB64 = aie::vector<int8,64>;
using VC16 = aie::vector<int8,16>;

constexpr int XC8 = XC / 8;
constexpr int YC8 = YC / 8;
constexpr int TAP_BYTES    = 64;
constexpr int B_stride_yc8 = KH * KW * TAP_BYTES;
constexpr int B_stride_xc8 = YC8 * B_stride_yc8;

constexpr int RB_ROWS = KH + SH;
alignas(32) static const int8 ZERO8[8] = {0};

// Pack TB16 from ring buffer; pad→zeros
static inline VA16 pack_tb16_from_rb(
  int abs_top, int xw, int xc8,
  int row_tags[RB_ROWS],
  int8 (&rb)[RB_ROWS][XW*XC])
{
  alignas(32) int8 tmp[16];

  if (abs_top >= 0 && abs_top < XH && xw >= 0 && xw < XW) {
    const int st = abs_top % RB_ROWS;
    if (row_tags[st] == abs_top) { __builtin_memcpy(&tmp[0], &rb[st][xw*XC + xc8*8], 8); }
    else                         { __builtin_memcpy(&tmp[0], ZERO8, 8); }
  } else                         { __builtin_memcpy(&tmp[0], ZERO8, 8); }

  const int abs_bot = abs_top + SH;
  if (abs_bot >= 0 && abs_bot < XH && xw >= 0 && xw < XW) {
    const int sb = abs_bot % RB_ROWS;
    if (row_tags[sb] == abs_bot) { __builtin_memcpy(&tmp[8], &rb[sb][xw*XC + xc8*8], 8); }
    else                         { __builtin_memcpy(&tmp[8], ZERO8, 8); }
  } else                         { __builtin_memcpy(&tmp[8], ZERO8, 8); }

  return aie::load_v<16>(tmp);
}

// ---- Row-major loader (PLIO) ----
static inline void load_row_major_until(
  input_stream_int32 * __restrict s_in32,
  int need_max,
  int  &rows_ready,
  int8 (&rb)[RB_ROWS][XW*XC],
  int   row_tags[RB_ROWS])
{
  while (rows_ready < need_max && (rows_ready + 1) < XH)
  chess_prepare_for_pipelining chess_loop_range(1,)
  {
    const int abs_h = rows_ready + 1;
    const int slot  = abs_h % RB_ROWS;
    int32* __restrict dst32 = reinterpret_cast<int32*>(&rb[slot][0]);
    const int words32 = (XW*XC) / 4;

    for (int w = 0; w < words32; ++w)
    chess_prepare_for_pipelining chess_loop_range(1,)
    {
      int32 r = readincr(s_in32);
      *dst32++ = r;
    }
    row_tags[slot] = abs_h;
    rows_ready = abs_h;
  }
}

// ---- TB16 loader (from previous Conv) ----
static inline void load_tb16_until(
  input_stream_int32 * __restrict s_in32,
  int need_max,
  int  &rows_ready,
  int  &tb_yh2, int &tb_yw, int &tb_yc8,     // state of incoming TB16 traversal
  int8 (&rb)[RB_ROWS][XW*XC],
  int   row_tags[RB_ROWS])
{
  while (rows_ready < need_max && tb_yh2 < XH)
  chess_prepare_for_pipelining chess_loop_range(1,)
  {
    // Read one TB16 = 4×int32 (LE): bytes[0..15]
    int32 r0 = readincr(s_in32);
    int32 r1 = readincr(s_in32);
    int32 r2 = readincr(s_in32);
    int32 r3 = readincr(s_in32);

    alignas(16) int32 w32[4];
    w32[0]=r0; w32[1]=r1; w32[2]=r2; w32[3]=r3;
    const int8* __restrict b = reinterpret_cast<const int8*>(w32);

    // Place top(0..7) into row tb_yh2, bottom(8..15) into row tb_yh2+1 (if exists)
    const int off = tb_yw*XC + tb_yc8*8;

    const int slot_t = tb_yh2 % RB_ROWS;
    __builtin_memcpy(&rb[slot_t][off], &b[0], 8);

    if ((tb_yh2 + 1) < XH) {
      const int slot_b = (tb_yh2 + 1) % RB_ROWS;
      __builtin_memcpy(&rb[slot_b][off], &b[8], 8);
    }

    // Advance TB16 cursor
    tb_yc8++;
    if (tb_yc8 == XC8) { tb_yc8 = 0; tb_yw++; }

    if (tb_yw == XW) {
      // Completed both rows in this pair
      row_tags[slot_t] = tb_yh2;
      rows_ready = tb_yh2;
      if ((tb_yh2 + 1) < XH) {
        const int slot_b = (tb_yh2 + 1) % RB_ROWS;
        row_tags[slot_b] = tb_yh2 + 1;
        rows_ready = tb_yh2 + 1;
      }
      tb_yw = 0; tb_yh2 += 2;
    }
  }
}

// ---- Main kernel ----
static inline void conv2d_i8_stream(
    input_stream_int32 * __restrict s_in32,
    output_stream_int32* __restrict s_out32)
{
  alignas(32) static int8 rb[RB_ROWS][XW*XC];
  int row_tags[RB_ROWS];
  for (int i = 0; i < RB_ROWS; ++i)
  chess_prepare_for_pipelining chess_loop_range(1,)
  { row_tags[i] = -1; }

  int rows_ready = -1;

  // TB16 traversal state (used only when IN_TB16=1)
  int tb_yh2 = 0, tb_yw = 0, tb_yc8 = 0;

  for (int yh2 = 0; yh2 < YH; yh2 += 2)
  chess_prepare_for_pipelining chess_loop_range(1,)
  {
    const int need_top_max = yh2*SH - PH + (KH-1);
    const int need_bot_max = need_top_max + SH;
    const int need_max     = need_bot_max;

#if IN_TB16
    load_tb16_until(s_in32, need_max, rows_ready, tb_yh2, tb_yw, tb_yc8, rb, row_tags);
#else
    load_row_major_until(s_in32, need_max, rows_ready, rb, row_tags);
#endif

    for (int yw = 0; yw < YW; ++yw)
    chess_prepare_for_pipelining chess_loop_range(1,)
    {
      for (int yc8 = 0; yc8 < YC8; ++yc8)
      chess_prepare_for_pipelining chess_loop_range(1,)
      {
        MM C; bool first = true;

        for (int xc8 = 0; xc8 < XC8; ++xc8)
        chess_prepare_for_pipelining chess_loop_range(1,)
        {
          const int8* __restrict pBci = k_p + xc8*B_stride_xc8 + yc8*B_stride_yc8;

          for (int kh = 0; kh < KH; ++kh)
          chess_prepare_for_pipelining chess_loop_range(1,)
          {
            const int xh_top = yh2*SH - PH + kh;

            for (int kw = 0; kw < KW; ++kw)
            chess_prepare_for_pipelining chess_loop_range(1,)
            {
              const int xw = yw*SW - PW + kw;
              VA16 Av = pack_tb16_from_rb(xh_top, xw, xc8, row_tags, rb);
              VB64 Bv = aie::load_v<64>(pBci);  pBci += TAP_BYTES;

              if (first) { C.mul(Av, Bv); first = false; }
              else       { C.mac(Av, Bv); }
            }
          }
        }

        // Quantize/ReLU, pack 16 bytes → 4×int32, zero bottom lane if tail
        alignas(16) int8 bytes16[16];
        VC16 Cv = C.template to_vector<int8>(SHIFT);
        if (DO_RELU) Cv = aie::max(Cv, (int8)0);
        aie::store_v(bytes16, Cv);

        if ((yh2 + 1) >= YH) {
          for (int i = 8; i < 16; ++i)
          chess_prepare_for_pipelining chess_loop_range(1,)
          { bytes16[i] = 0; }
        }

        const int32* p32 = reinterpret_cast<const int32*>(bytes16);
        for (int i = 0; i < 4; ++i)
        chess_prepare_for_pipelining chess_loop_range(1,)
        { writeincr(s_out32, p32[i]); }
      }
    }
  }
}
