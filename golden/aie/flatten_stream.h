#include <adf.h>
#include <cstdint>
#include "aie_api/aie.hpp"
#include "aie_api/aie_adf.hpp"

struct FlatReader {
  input_stream_int8* __restrict sin;
  alignas(16) int8 buf[16];
  int avail;
  int idx;
  FlatReader(input_stream_int8* __restrict s) : sin(s), avail(0), idx(0) {}
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

struct FlatWriter {
  output_stream_int8* __restrict sout;
  alignas(16) int8 buf[16];
  int fill;
  FlatWriter(output_stream_int8* __restrict s) : sout(s), fill(0) {}
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

static inline void flatten_stream(
    input_stream_int8* __restrict s_in,
    output_stream_int8* __restrict s_out)
{
  FlatReader reader(s_in);
  FlatWriter writer(s_out);

  const int total = FLAT_IN_H * FLAT_IN_W * FLAT_IN_C;
  alignas(32) int8 flat_buf[FLAT_KREAL];
  for (int i = 0; i < total; ++i) {
    flat_buf[i] = reader.pop();
  }

  const int br_total = FLAT_RPAD / FLAT_TM;
  const int bc_total = FLAT_KPAD / FLAT_TK;

  for (int br = 0; br < br_total; ++br) {
    for (int bc = 0; bc < bc_total; ++bc) {
      for (int r = 0; r < FLAT_TM; ++r) {
        for (int c = 0; c < FLAT_TK; ++c) {
          const int row = br * FLAT_TM + r;
          const int col = bc * FLAT_TK + c;
          int8 val = 0;
          if (row < FLAT_RREAL && col < FLAT_KREAL) {
            val = flat_buf[row * FLAT_KREAL + col];
          }
          writer.push(val);
        }
      }
    }
  }

  writer.finalize();
}
