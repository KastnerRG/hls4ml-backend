#include <adf.h>
#include <cstdint>
#include "aie_api/aie.hpp"
#include "aie_api/aie_adf.hpp"

struct StreamReaderRS {
  input_stream_int8* __restrict sin;
  alignas(16) int8 buf[16];
  int avail;
  int idx;
  StreamReaderRS(input_stream_int8* __restrict s) : sin(s), avail(0), idx(0) {}
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

struct StreamWriterRS {
  output_stream_int8* __restrict sout;
  alignas(16) int8 buf[16];
  int fill;
  StreamWriterRS(output_stream_int8* __restrict s) : sout(s), fill(0) {}
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

static inline void reshape_stream(
    input_stream_int8* __restrict s_in,
    output_stream_int8* __restrict s_out)
{
  StreamReaderRS reader(s_in);
  StreamWriterRS writer(s_out);

  alignas(32) int8 buffer[PREV_RREAL * PREV_CO];

  const int prev_br = PREV_RPAD / PREV_TM;
  const int prev_bc = PREV_COPAD / PREV_TN;
  for (int br = 0; br < prev_br; ++br) {
    for (int bc = 0; bc < prev_bc; ++bc) {
      for (int r = 0; r < PREV_TM; ++r) {
        for (int c = 0; c < PREV_TN; ++c) {
          const int row = br * PREV_TM + r;
          const int col = bc * PREV_TN + c;
          int8 v = reader.pop();
          if (row < PREV_RREAL && col < PREV_CO) {
            buffer[row * PREV_CO + col] = v;
          }
        }
      }
    }
  }

  const int next_br = NEXT_RPAD / NEXT_TM;
  const int next_bc = NEXT_KPAD / NEXT_TN;

  for (int br = 0; br < next_br; ++br) {
    for (int bc = 0; bc < next_bc; ++bc) {
      for (int r = 0; r < NEXT_TM; ++r) {
        for (int c = 0; c < NEXT_TN; ++c) {
          const int row = br * NEXT_TM + r;
          const int col = bc * NEXT_TN + c;
          int8 val = 0;
          if (row < NEXT_RREAL && col < NEXT_KREAL) {
            const int oh = row / NEXT_YW;
            const int ow = row % NEXT_YW;
            int kidx = col;
            const int ci = kidx % NEXT_CI;
            kidx /= NEXT_CI;
            const int kw = kidx % NEXT_KW;
            const int kh = kidx / NEXT_KW;
            const int ih = oh * NEXT_SH - NEXT_PH + kh;
            const int iw = ow * NEXT_SW - NEXT_PW + kw;
            if (ih >= 0 && ih < PREV_YH && iw >= 0 && iw < PREV_YW) {
              val = buffer[(ih * PREV_YW + iw) * PREV_CO + ci];
            }
          }
          writer.push(val);
        }
      }
    }
  }

  writer.finalize();
}
