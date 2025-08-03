#ifndef FUNCTION_INCLUDES_H
#define FUNCTION_INCLUDES_H

// define shift right for output values after matrix mult
#define SHIFT 0

// single kernel dimensions (MxKxN on manuscript)
#define single_M 16
#define single_K 32
#define single_N 16


// AI Engine API dimensions
#define M_API 4
#define K_API 8
#define N_API 4

// INT8 sizes
// 4x8x4
// 4x16x4
// 8x8x4
// 2x8x8
// 4x8x8
// 2x16x8
// 4x16x8

#endif
