#include <adf.h>
#include <cstdint>
#include "aie_api/aie.hpp"
#include "aie_api/aie_adf.hpp"

#ifdef chess_prepare_for_pipelining
#undef chess_prepare_for_pipelining
#define chess_prepare_for_pipelining
#endif
#ifdef chess_loop_range
#undef chess_loop_range
#define chess_loop_range(...)
#endif

using MM   = aie::mmul<2, 8, 8, int8, int8>;
using VA16 = aie::vector<int8, 16>;
using VB64 = aie::vector<int8, 64>;
using VC16 = aie::vector<int8, 16>;

#ifndef DO_RELU
#define DO_RELU false
#endif
#ifndef PAD_IN
#define PAD_IN 0
#endif
#ifndef PAD_OUT
#define PAD_OUT 0
#endif

extern const int8_t k_p[];

constexpr int XC8 = (CI + 7) / 8;
constexpr int YC8 = (CO + 7) / 8;
constexpr int CO_ALIGNED = YC8 * 8;
static_assert(KH > 0 && KW > 0, "Kernel dimensions must be positive");
static_assert(SH > 0 && SW > 0, "Strides must be positive");

constexpr int TAP_BYTES    = 64; // 8 (ci lanes) * 8 (co lanes)
constexpr int B_stride_yc8 = KH * KW * TAP_BYTES;
constexpr int B_stride_xc8 = YC8 * B_stride_yc8;
constexpr int RB_ROWS      = KH + SH;


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

struct StreamReader {
  input_stream_int8* __restrict sin;
  alignas(16) int8 buf[16];
  int avail;
  int idx;

  StreamReader(input_stream_int8* __restrict s) : sin(s), avail(0), idx(0) {}

  inline void refill() {
    auto v = readincr_v<16>(sin);
    aie::store_v(buf, v);
    avail = 16;
    idx = 0;
  }

  inline int8 pop() {
    if (avail == 0) { refill(); }
    int8 val = buf[idx++];
    avail--;
    return val;
  }

  inline void copy(int8* __restrict dst, int bytes) {
    for (int i = 0; i < bytes; ++i) {
      dst[i] = pop();
    }
  }

  inline void skip(int bytes) {
    for (int i = 0; i < bytes; ++i) {
      (void)pop();
    }
  }
};

struct StreamWriter {
  output_stream_int8* __restrict sout;
  alignas(16) int8 buf[16];
  int fill;

  StreamWriter(output_stream_int8* __restrict s) : sout(s), fill(0) {}

  inline void flush_full() {
    auto v = aie::load_v<16>(buf);
    writeincr(sout, v);
    fill = 0;
  }

  inline void push(int8 val) {
    buf[fill++] = val;
    if (fill == 16) flush_full();
  }

  inline void write(const int8* __restrict src, int bytes) {
    for (int i = 0; i < bytes; ++i) push(src[i]);
  }

  inline void pad(int bytes) {
    for (int i = 0; i < bytes; ++i) push((int8)0);
  }

  inline void finalize() {
    if (fill) {
      while (fill < 16) buf[fill++] = 0;
      flush_full();
    }
  }
};

static inline void load_rows_until(
    StreamReader &reader,
    int need_max,
    int &rows_ready,
    int8 (&rb)[RB_ROWS][XW * CI],
    int  row_tags[RB_ROWS])
{
  const int row_bytes = XW * CI;

  while (rows_ready < need_max && (rows_ready + 1) < XH)
  chess_prepare_for_pipelining chess_loop_range(1,)
  {
    const int abs_h = rows_ready + 1;
    const int slot  = abs_h % RB_ROWS;
    int8* __restrict dst = &rb[slot][0];

    reader.copy(dst, row_bytes);

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
  StreamReader reader(s_in);
  StreamWriter writer(s_out);

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

      load_rows_until(reader, need_max, rows_ready, rb, row_tags);

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

          const int channel_base = yc8 * 8;
          int valid = CO - channel_base;
          if (valid > 8) valid = 8;
          if (valid > 0) {
            const int dst_off_top = yw * CO + channel_base;
            __builtin_memcpy(&row_buf[0][dst_off_top], &bytes16[0], valid);
          }
          if ((yh2 + 1) < YH && valid > 0) {
            const int dst_off_bot = yw * CO + channel_base;
            __builtin_memcpy(&row_buf[1][dst_off_bot], &bytes16[8], valid);
          }
        }
      }

      for (int r = 0; r < 2; ++r)
      chess_prepare_for_pipelining chess_loop_range(1,)
      {
        if ((yh2 + r) >= YH) break;
        writer.write(&row_buf[r][0], YW * CO);
      }
    }

    if (PAD_IN > 0) {
      reader.skip(PAD_IN);
    }

    if (PAD_OUT > 0) {
      writer.pad(PAD_OUT);
    }
    writer.finalize();

#ifdef FREE
  }
#endif
}
