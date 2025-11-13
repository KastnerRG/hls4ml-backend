#include <adf.h>
#include <cstdint>
#include "aie_api/aie.hpp"
#include "aie_api/aie_adf.hpp"

struct StreamReaderFR {
  input_stream_int8* __restrict sin;
  alignas(16) int8 buf[16];
  int avail;
  int idx;
  StreamReaderFR(input_stream_int8* __restrict s) : sin(s), avail(0), idx(0) {}
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

struct StreamWriterFR {
  output_stream_int8* __restrict sout;
  alignas(16) int8 buf[16];
  int fill;
  StreamWriterFR(output_stream_int8* __restrict s) : sout(s), fill(0) {}
  inline void flush_full() {
    auto v = aie::load_v<16>(buf);
    writeincr(sout, v);
    fill = 0;
  }
  inline void push(int8 val) {
    buf[fill++] = val;
    if (fill == 16) flush_full();
  }
  inline void pad_zeros(int count) {
    for (int i = 0; i < count; ++i) push(0);
  }
  inline void finalize() {
    if (fill) {
      while (fill < 16) buf[fill++] = 0;
      flush_full();
    }
  }
};

static inline void reshape_to_nhwc(
    input_stream_int8* __restrict s_in,
    output_stream_int8* __restrict s_out)
{
  StreamReaderFR reader(s_in);
  StreamWriterFR writer(s_out);

  alignas(32) int8 buffer[FINAL_RREAL * FINAL_CO];

  const int br_total = FINAL_RPAD / FINAL_TM;
  const int bc_total = FINAL_COPAD / FINAL_TN;

  for (int br = 0; br < br_total; ++br) {
    for (int bc = 0; bc < bc_total; ++bc) {
      for (int r = 0; r < FINAL_TM; ++r) {
        for (int c = 0; c < FINAL_TN; ++c) {
          const int row = br * FINAL_TM + r;
          const int col = bc * FINAL_TN + c;
          int8 v = reader.pop();
          if (row < FINAL_RREAL && col < FINAL_CO) {
            buffer[row * FINAL_CO + col] = v;
          }
        }
      }
    }
  }

  for (int r = 0; r < FINAL_RREAL; ++r) {
    for (int c = 0; c < FINAL_CO; ++c) {
      writer.push(buffer[r * FINAL_CO + c]);
    }
  }

  writer.pad_zeros(FINAL_PAD);
  writer.finalize();
}
