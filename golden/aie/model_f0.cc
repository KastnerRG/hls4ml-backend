#include "kernels.h"
#include "weights.h"
void f0(input_window_int8* __restrict x, output_window_int8 * __restrict a) { conv2d_v_tiny<4,4,8,8,1,1>(x, a, k0, 2, false); }
