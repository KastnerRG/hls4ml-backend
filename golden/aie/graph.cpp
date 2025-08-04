
#include <adf.h>
#include "kernels.h"
#include "include.h"
#include <vector>

using namespace adf;

class simpleGraph : public adf::graph {
private:
  kernel mat_mul_k;

public:

  input_plio  A;
  output_plio C;

  simpleGraph(){

    A = input_plio::create(plio_128_bits, "data/matA0.txt");
    C = output_plio::create(plio_128_bits, "data/matC0.txt");

    mat_mul_k = kernel::create(gemm);

    connect< window<M*K*1> >  (A.out[0], mat_mul_k.in[0]);
    connect< window<M*N*4> >  (mat_mul_k.out[0], C.in[0]);

    source(mat_mul_k) = "kernels.cc";
    runtime<ratio>(mat_mul_k) = 1.0;
  }
};

simpleGraph mygraph;

int main(void) {
  mygraph.init();
  mygraph.run(B);
  mygraph.end();
  return 0;
}
