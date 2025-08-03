
#include <vector>

#include <adf.h>
#include "kernels.h"
#include "kernels/include.h"

using namespace adf;

class simpleGraph : public adf::graph {
private:


  kernel mat_mul_k;

public:

  input_plio  A;
  input_plio  B;
  output_plio C;


  simpleGraph(){

	  // input and output PLIOs creation below
		A = input_plio::create(plio_128_bits, "data/matA0.txt");
		B = input_plio::create(plio_128_bits, "data/matB0.txt");
		C = output_plio::create(plio_128_bits, "data/matC0.txt");

	  // kernels creation
	  mat_mul_k = kernel::create(gemm);

	  // Single kernel connections
	  connect< window<M*K*1> >  (A.out[0], mat_mul_k.in[0]);
	  connect< window<K*N*1> >  (B.out[0], mat_mul_k.in[1]);

	  // Place buffers in different banks to prevent memory stalls (see UG1076 for more details)
	  not_equal(location<buffer>(mat_mul_k.in[0]), location<buffer>(mat_mul_k.in[1]));

	  connect< window<M*N*4> >  (mat_mul_k.out[0], C.in[0]);

	  // direct the source file of kernels
	  source(mat_mul_k) = "kernels/kernels.cc";

	  runtime<ratio>(mat_mul_k) = 1.0;
  }
};
