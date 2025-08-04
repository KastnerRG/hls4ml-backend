
#ifndef FUNCTION_KERNELS_H
#define FUNCTION_KERNELS_H

#include <adf.h>
#include "aie_api/aie.hpp"
#include "aie_api/aie_adf.hpp"


template <int m, int k, int n, int M, int K, int N, int SHIFT>
void gemm(
	input_window_int8 * __restrict matA, 
	output_window_int8 * __restrict matC,
	const int8 matB []
	){
  using MMUL = aie::mmul<m, k, n, int8, int8>;

	const int num_rowA = (M/m);
	const int num_colA = (K/k);
	const int num_colB = (N/n);

  const int8* __restrict pA=(int8*)matA->ptr;
  const int8* __restrict pB=(int8*)matB;
  int8* __restrict pC = (int8*) matC->ptr;

  //For profiling only 
  unsigned long long cycle_num[2];
  aie::tile tile=aie::tile::current();
  cycle_num[0]=tile.cycles();

  for (unsigned ii = 0; ii < num_rowA; ++ii) 
  chess_unroll_loop(num_rowA)
  {
    for (unsigned jj = 0; jj < num_colB; ++jj) 
    chess_unroll_loop(num_colB)
    {
      const int8 * __restrict pA1 = pA + ( ii * num_colA + 0) * MMUL::size_A;
      const int8 * __restrict pB1 = pB + ( 0 * num_colB + jj) * MMUL::size_B;

      aie::vector<int8, MMUL::size_A> A0 = aie::load_v<MMUL::size_A>(pA1); pA1 += MMUL::size_A;
      aie::vector<int8, MMUL::size_B> B0 = aie::load_v<MMUL::size_B>(pB1); pB1 += MMUL::size_B * num_colB;

      MMUL C00; C00.mul(A0, B0);

      for (unsigned kk = 0; kk < num_colA-1; ++kk) 
      chess_flatten_loop
      {
        A0 = aie::load_v<MMUL::size_A>(pA1); pA1 += MMUL::size_A;
        B0 = aie::load_v<MMUL::size_B>(pB1); pB1 += MMUL::size_B * num_colB;
        C00.mac(A0, B0);
      }
			auto C00_vec = C00.template to_vector<int8>(SHIFT);
			auto C00_relu = aie::max(C00_vec, (int8)0);
      aie::store_v(pC, C00_relu); 
			pC += MMUL::size_C;
    }
  }
  //For profiling only 
  cycle_num[1]=tile.cycles();
  printf("start=%lld,end=%lld,total=%lld\n",cycle_num[0],cycle_num[1],cycle_num[1]-cycle_num[0]);
}

#endif