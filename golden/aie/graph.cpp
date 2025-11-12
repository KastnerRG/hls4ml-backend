#include <adf.h>
#include "include.h"
#include <cstdio>
#include <cstdlib>
#include <sys/stat.h>
#include <chrono>
#include <thread>

using namespace adf;

class simpleGraph : public graph {
  kernel layers[N_LAYERS];
public:
  input_plio  AIE_IN;
  output_plio AIE_OUT;

  simpleGraph(){
    AIE_IN  = input_plio::create(plio_128_bits, "../../data/x0.txt");
    AIE_OUT = output_plio::create(plio_128_bits, "data/out_sim.txt");
    #include "layer_graph.h"
    for (int i = 0; i < N_LAYERS; i++) runtime<ratio>(layers[i]) = 1.0;
  }
};

simpleGraph mygraph;

int main() {
  mygraph.init();

  auto h = adf::event::start_profiling(
      mygraph.AIE_IN, mygraph.AIE_OUT,
      adf::event::io_stream_start_difference_cycles);

#ifdef FREE
  mygraph.run(-1);
#else
  mygraph.run(ITERATIONS);
  mygraph.end();
#endif

  long long cyc = adf::event::read_profiling(h);
  adf::event::stop_profiling(h);

  const double AIE_clk = 1.2e9;
  printf("\n\n Graph Latency (first->first): %lld cycles, %.1f ns\n\n", cyc, (1e9 * cyc) / AIE_clk);

  return 0;
}
