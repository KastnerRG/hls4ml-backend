#include <adf.h>
#include "aie_api/aie.hpp"
#include "aie_api/aie_adf.hpp"
#include "include.h"

#define num_rowA (single_M/M_API)
#define num_colA (single_K/K_API)
#define num_colB (single_N/N_API)

void gemm(input_window_int8 * __restrict matA, input_window_int8 * __restrict matB, output_window_int32 * __restrict matC){
  using MMUL = aie::mmul<M_API, K_API, N_API, int8, int8>;

  const int8* __restrict pA=(int8*)matA->ptr;
  const int8* __restrict pB=(int8*)matB->ptr;
  int32* __restrict pC=(int32*)matC->ptr;

  //For profiling only 
  unsigned long long cycle_num[2];
  aie::tile tile=aie::tile::current();
  cycle_num[0]=tile.cycles();

  for (unsigned i = 0; i < num_rowA; ++i) 
  chess_unroll_loop(num_rowA)
  {
    for (unsigned j = 0; j < num_colB; ++j) 
    chess_unroll_loop(num_colB)
    {
      const int8 * __restrict pA1 = pA + ( i * num_colA + 0) * MMUL::size_A;
      const int8 * __restrict pB1 = pB + ( 0 * num_colB + j) * MMUL::size_B;

      aie::vector<int8, MMUL::size_A> A0 = aie::load_v<MMUL::size_A>(pA1); pA1 += MMUL::size_A;
      aie::vector<int8, MMUL::size_B> B0 = aie::load_v<MMUL::size_B>(pB1); pB1 += MMUL::size_B * num_colB;

      MMUL C00; C00.mul(A0, B0);

      for (unsigned k = 0; k < num_colA-1; ++k) 
      chess_flatten_loop
      {
        A0 = aie::load_v<MMUL::size_A>(pA1); pA1 += MMUL::size_A;
        B0 = aie::load_v<MMUL::size_B>(pB1); pB1 += MMUL::size_B * num_colB;
        C00.mac(A0, B0);
      }
      aie::store_v(pC, C00.template to_vector<int32>(SHIFT)); pC += MMUL::size_C;
    }
  }
  //For profiling only 
  cycle_num[1]=tile.cycles();
  printf("start=%lld,end=%lld,total=%lld\n",cycle_num[0],cycle_num[1],cycle_num[1]-cycle_num[0]);
}