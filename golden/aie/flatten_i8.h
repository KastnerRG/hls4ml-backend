#include <adf.h>
#include "aie_api/aie.hpp"
#include "aie_api/aie_adf.hpp"
using namespace adf;

void flatten_i8(input_window_int8  * __restrict in_nhwc,
                output_window_int8 * __restrict out_hw_by_c)
{
  // NHWC is already contiguous in C-major; (H*W,C) is the same linear order.
  const int8 * __restrict in  = (const int8*)in_nhwc->ptr;
  int8       * __restrict out = (int8*)      out_hw_by_c->ptr;

  // Copy as is (window sizes ensure exact length)
  // Use 128-bit moves for speed
  constexpr int V = 16;
  const int total_bytes = window_size(in_nhwc);
  int i = 0;
  for (; i + V <= total_bytes; i += V) {
    aie::vector<int8, V> v = aie::load_v<V>(in + i);
    aie::store_v(out + i, v);
  }
  for (; i < total_bytes; ++i) out[i] = in[i];
}