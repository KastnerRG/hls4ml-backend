#include <adf.h>
#include "aie_api/aie.hpp"
#include "aie_api/aie_adf.hpp"
#include "kernels.h"
#include "weights.h"

#define DENSE_FN(IDX, API_M, API_K, API_N, S_M, S_K, S_N)               \
  void f##IDX(  input_window_int8  * __restrict matA,                   \
                output_window_int8 * __restrict matC) {                 \
    gemm<API_M, API_K, API_N, S_M, S_K, S_N, 0>(matA, matC, matB##IDX); \
  }

DENSE_FN(0, 4, 8, 4, 16, 32, 16)

#undef DENSE_FN