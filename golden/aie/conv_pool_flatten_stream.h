#include <adf.h>
#include <cstdint>
#include "aie_api/aie.hpp"
#include "aie_api/aie_adf.hpp"

struct CPFReader {
  input_stream_int8* __restrict sin;
  alignas(16) int8 buf[16];
  int avail;
  int idx;
  CPFReader(input_stream_int8* __restrict s) : sin(s), avail(0), idx(0) {}
  inline int8 pop() {
    if (avail == 0) {
      auto v = readincr_v<16>(sin);
      aie::store_v(buf, v);
      avail = 16;
      idx = 0;
    }
    avail--;
    return buf[idx++];
  }
};

struct CPFWriter {
  output_stream_int8* __restrict sout;
  alignas(16) int8 buf[16];
  int fill;
  CPFWriter(output_stream_int8* __restrict s) : sout(s), fill(0) {}
  inline void flush_full() {
    auto v = aie::load_v<16>(buf);
    writeincr(sout, v);
    fill = 0;
  }
  inline void push(int8 val) {
    buf[fill++] = val;
    if (fill == 16) flush_full();
  }
  inline void finalize() {
    if (fill) {
      while (fill < 16) buf[fill++] = 0;
      flush_full();
    }
  }
};

static inline void conv_pool_flatten_stream(
    input_stream_int8* __restrict s_in,
    output_stream_int8* __restrict s_out)
{
  CPFReader reader(s_in);
  CPFWriter writer(s_out);

  const int prev_rows = CPF_PREV_RREAL;
  const int prev_cols = CPF_PREV_CO;
  const int rows_per_img = CPF_PREV_YH * CPF_PREV_YW;
  const int prev_br = CPF_PREV_RPAD / CPF_TM;
  const int prev_bc = CPF_PREV_COPAD / CPF_TN;

  alignas(32) static int8 prev_buf[CPF_PREV_RREAL * CPF_PREV_CO];
  for (int br = 0; br < prev_br; ++br) {
    for (int bc = 0; bc < prev_bc; ++bc) {
      for (int r = 0; r < CPF_TM; ++r) {
        for (int c = 0; c < CPF_TN; ++c) {
          const int row = br * CPF_TM + r;
          const int col = bc * CPF_TN + c;
          int8 v = reader.pop();
          if (row < prev_rows && col < prev_cols) {
            prev_buf[row * prev_cols + col] = v;
          }
        }
      }
    }
  }

  const int pool_area = CPF_POOL_KH * CPF_POOL_KW;
  const int flat_rows = CPF_FLAT_RREAL;
  const int flat_cols = CPF_FLAT_KREAL;

  alignas(32) static int8 flat_buf[CPF_FLAT_RREAL * CPF_FLAT_KREAL];

  for (int b = 0; b < CPF_BATCH; ++b) {
    int flat_idx = 0;
    for (int oh = 0; oh < CPF_POOL_OUT_H; ++oh) {
      for (int ow = 0; ow < CPF_POOL_OUT_W; ++ow) {
        for (int ch = 0; ch < CPF_PREV_CO; ++ch) {
          int32 acc = 0;
          for (int kh = 0; kh < CPF_POOL_KH; ++kh) {
            int ih = oh * CPF_POOL_SH + kh - CPF_POOL_PH;
            for (int kw = 0; kw < CPF_POOL_KW; ++kw) {
              int iw = ow * CPF_POOL_SW + kw - CPF_POOL_PW;
              if (ih >= 0 && ih < CPF_PREV_YH && iw >= 0 && iw < CPF_PREV_YW) {
                const int row = b * rows_per_img + ih * CPF_PREV_YW + iw;
                acc += prev_buf[row * prev_cols + ch];
              }
            }
          }
          acc /= pool_area;
          if (acc > 127) acc = 127;
          if (acc < -128) acc = -128;
          flat_buf[b * flat_cols + flat_idx++] = (int8)acc;
        }
      }
    }
  }

  const int br_total = CPF_FLAT_RPAD / CPF_FLAT_TM;
  const int bc_total = CPF_FLAT_KPAD / CPF_FLAT_TK;
  for (int br = 0; br < br_total; ++br) {
    for (int bc = 0; bc < bc_total; ++bc) {
      for (int r = 0; r < CPF_FLAT_TM; ++r) {
        for (int c = 0; c < CPF_FLAT_TK; ++c) {
          const int row = br * CPF_FLAT_TM + r;
          const int col = bc * CPF_FLAT_TK + c;
          int8 val = 0;
          if (row < flat_rows && col < flat_cols) {
            val = flat_buf[row * flat_cols + col];
          }
          writer.push(val);
        }
      }
    }
  }

  writer.finalize();
}
