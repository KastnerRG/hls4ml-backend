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

  alignas(32) static int8 prev_buf[CPF_PREV_RREAL * CPF_PREV_CO];
  const int prev_br = CPF_PREV_RPAD / CPF_TM;
  const int prev_bc = CPF_PREV_COPAD / CPF_TN;
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
  const int rows_per_group = CPF_FLAT_TM;
  const int cols_per_group = CPF_FLAT_TK;
  const int row_groups = CPF_FLAT_RPAD / CPF_FLAT_TM;
  const int col_groups = CPF_FLAT_KPAD / CPF_FLAT_TK;
  constexpr int TILE_ELEMS = CPF_FLAT_TM * CPF_FLAT_TK;
  static_assert(TILE_ELEMS == 16, "ConvPoolFlatten assumes 2x8 tile output");

  alignas(32) int8 row_tile[CPF_FLAT_TM][CPF_FLAT_KPAD];
  alignas(32) int8 tile_buf[TILE_ELEMS];

  for (int br = 0; br < row_groups; ++br) {
    for (int r = 0; r < rows_per_group; ++r) {
      int row_idx = br * rows_per_group + r;
      int8* row_ptr = row_tile[r];
      for (int i = 0; i < CPF_FLAT_KPAD; ++i) {
        row_ptr[i] = 0;
      }
      if (row_idx < CPF_BATCH) {
        int flat_idx = 0;
        const int batch_offset = row_idx * rows_per_img;
        for (int oh = 0; oh < CPF_POOL_OUT_H; ++oh) {
          const int base_ih = oh * CPF_POOL_SH - CPF_POOL_PH;
          for (int ow = 0; ow < CPF_POOL_OUT_W; ++ow) {
            const int base_iw = ow * CPF_POOL_SW - CPF_POOL_PW;
            for (int ch = 0; ch < CPF_PREV_CO; ++ch) {
              int32 acc = 0;
              for (int kh = 0; kh < CPF_POOL_KH; ++kh) {
                int ih = base_ih + kh;
                if (ih < 0 || ih >= CPF_PREV_YH) continue;
                for (int kw = 0; kw < CPF_POOL_KW; ++kw) {
                  int iw = base_iw + kw;
                  if (iw < 0 || iw >= CPF_PREV_YW) continue;
                  const int row = batch_offset + ih * CPF_PREV_YW + iw;
                  acc += prev_buf[row * prev_cols + ch];
                }
              }
              acc /= pool_area;
              if (acc > 127) acc = 127;
              if (acc < -128) acc = -128;
              if (flat_idx < CPF_FLAT_KREAL) {
                row_ptr[flat_idx++] = (int8)acc;
              }
            }
          }
        }
      }
    }

    for (int bc = 0; bc < col_groups; ++bc) {
      for (int r = 0; r < rows_per_group; ++r) {
        for (int c = 0; c < cols_per_group; ++c) {
          tile_buf[r * cols_per_group + c] =
              row_tile[r][bc * cols_per_group + c];
        }
      }
      auto tile_vec = aie::load_v<TILE_ELEMS>(tile_buf);
      writeincr(s_out, tile_vec);
    }
  }

  writer.finalize();
}
