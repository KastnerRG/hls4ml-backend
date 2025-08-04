
#include <adf.h>
#include "include.h"
#include <vector>
#include "model.h"

using namespace adf;

class simpleGraph : public adf::graph {
private:
  kernel layers[N_LAYERS];

public:

  input_plio  A;
  output_plio C;

  simpleGraph(){

    A = input_plio::create(plio_128_bits, "data/matA0.txt");
    C = output_plio::create(plio_128_bits, "data/matC0.txt");

    layers[0] = kernel::create(f0);

    connect< window<M*K*1> >  (A.out[0], layers[0].in[0]);
    connect< window<M*N*1> >  (layers[0].out[0], C.in[0]);

    for (int i = 0; i < N_LAYERS; i++) {
      source(layers[i]) = "model.cc";
      runtime<ratio>(layers[i]) = 1.0;
    }
  }
};

simpleGraph mygraph;

int main(void) {
  mygraph.init();
  mygraph.run(ITERATIONS);
  mygraph.end();
  return 0;
}
