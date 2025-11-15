#include <adf.h>
#include "include.h"
#include <cstdio>
#include <cstdlib>
#include <sys/stat.h>
#include <chrono>
#include <thread>
#include <string>

using namespace adf;

class simpleGraph : public graph {
  kernel layers[N_LAYERS];
public:
  input_plio  AIE_IN[NUM_INPUT_PLIO];
  output_plio AIE_OUT[NUM_OUTPUT_PLIO];

  simpleGraph(){
    for (int i = 0; i < NUM_INPUT_PLIO; ++i) {
      std::string in_path = (NUM_INPUT_PLIO == 1)
          ? "../../data/x0.txt"
          : ("../../data/x0_" + std::to_string(i) + ".txt");
      AIE_IN[i] = input_plio::create(plio_128_bits, in_path.c_str());
    }
    for (int i = 0; i < NUM_OUTPUT_PLIO; ++i) {
      std::string out_path = (NUM_OUTPUT_PLIO == 1)
          ? "data/out_sim.txt"
          : ("data/out_sim_" + std::to_string(i) + ".txt");
      AIE_OUT[i] = output_plio::create(plio_128_bits, out_path.c_str());
    }
    #include "layer_graph.h"
    for (int i = 0; i < N_LAYERS; i++) runtime<ratio>(layers[i]) = 1.0;
  }
};

simpleGraph mygraph;

int main() {
  mygraph.init();

  auto h = adf::event::start_profiling(
      mygraph.AIE_IN[0], mygraph.AIE_OUT[0],
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
