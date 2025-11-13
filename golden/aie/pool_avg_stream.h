#include <adf.h>
#include <cstdint>
#include "aie_api/aie.hpp"
#include "aie_api/aie_adf.hpp"

#ifndef SHIFT
#define SHIFT 0
#endif
#ifndef DO_RELU
#define DO_RELU false
#endif
#ifndef POOL_BATCH
#define POOL_BATCH 1
#endif

struct PoolReader {
  input_stream_int8* __restrict sin;
  alignas(16) int8 buf[16];
  int avail;
  int idx;
  PoolReader(input_stream_int8* __restrict s) : sin(s), avail(0), idx(0) {}
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

struct PoolWriter {
  output_stream_int8* __restrict sout;
  alignas(16) int8 buf[16];
  int fill;
  PoolWriter(output_stream_int8* __restrict s) : sout(s), fill(0) {}
  inline void flush_full() {
    auto v = aie::load_v<16>(buf);
    writeincr(sout, v);
    fill = 0;
  }
  inline void push(int8 val) {
    buf[fill++] = val;
    if (fill == 16) flush_full();
  }
  inline void pad(int count) {
    for (int i = 0; i < count; ++i) push(0);
  }
  inline void finalize() {
    if (fill) {
      while (fill < 16) buf[fill++] = 0;
      flush_full();
    }
  }
};

static inline void pool_avg_stream(
    input_stream_int8* __restrict s_in,
    output_stream_int8* __restrict s_out)
{
  PoolReader reader(s_in);
  PoolWriter writer(s_out);

  const int per_img = POOL_IN_H * POOL_IN_W * POOL_C;
  alignas(32) static int8 buffer[POOL_BATCH * per_img];

  for (int b = 0; b < POOL_BATCH; ++b) {
    for (int i = 0; i < per_img; ++i) {
      buffer[b * per_img + i] = reader.pop();
    }
  }

  const int kernel_area = POOL_KH * POOL_KW;

  for (int b = 0; b < POOL_BATCH; ++b) {
    for (int oh = 0; oh < POOL_OUT_H; ++oh) {
      for (int ow = 0; ow < POOL_OUT_W; ++ow) {
        for (int c = 0; c < POOL_C; ++c) {
          int32 acc = 0;
          for (int kh = 0; kh < POOL_KH; ++kh) {
            int ih = oh * POOL_SH + kh;
            for (int kw = 0; kw < POOL_KW; ++kw) {
              int iw = ow * POOL_SW + kw;
              acc += buffer[b * per_img + (ih * POOL_IN_W + iw) * POOL_C + c];
            }
          }
          acc = acc / kernel_area;
          if (SHIFT > 0) {
            acc = (acc + (1 << (SHIFT - 1))) >> SHIFT;
          }
          if (DO_RELU && acc < 0) acc = 0;
          if (acc > 127) acc = 127;
          if (acc < -128) acc = -128;
          writer.push((int8)acc);
        }
      }
    }
  }

  writer.pad(POOL_PAD);
  writer.finalize();
}
