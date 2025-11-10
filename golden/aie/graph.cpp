#include <adf.h>
#include "include.h"
#include <vector>

using namespace adf;

class simpleGraph : public adf::graph {
private:
  kernel layers [N_LAYERS];

public:
  input_plio  AIE_IN;
  output_plio AIE_OUT;

  simpleGraph(){
    AIE_IN  = input_plio::create(plio_128_bits, "data/x0.txt");
    AIE_OUT = output_plio::create(plio_128_bits, "data/out_sim.txt");

    #include "layer_graph.h"

    for (int i = 0; i < N_LAYERS; i++) runtime<ratio>(layers[i]) = 1.0;
  }
};

simpleGraph mygraph;

int main() {
  constexpr double AIE_clock_Hz = 1.2e9;  // adjust if needed

  std::vector<long long> lat_cycles;

  for (int i = 0; i < ITERATIONS; ++i) {
    mygraph.init();  // cold start: no residual data in the pipes

    auto h = adf::event::start_profiling(
        mygraph.AIE_IN, mygraph.AIE_OUT,
        adf::event::io_stream_start_difference_cycles);

    mygraph.run(1);
    mygraph.wait();                 // ensure this iteration fully completes
    long long cyc = adf::event::read_profiling(h);
    adf::event::stop_profiling(h);

    mygraph.end();                  // drain/teardown -> next loop starts cold
    lat_cycles.push_back(cyc);
  }
  // If you want to print all measurements:
  for (int i = 0; i < (int)lat_cycles.size(); ++i) {
    double ns = (1e9 * (double)lat_cycles[i]) / AIE_clock_Hz;
    std::printf("Run %3d : %lld cycles, %.3f ns\n", i, lat_cycles[i], ns);
  }

  // Discard the first (warm-up) and compute steady-state stats
  if (lat_cycles.size() >= 2) {
    long long min_c = std::numeric_limits<long long>::max();
    long long max_c = 0;
    double sum = 0.0;

    for (size_t i = 1; i < lat_cycles.size(); ++i) {
      sum += (double)lat_cycles[i];
      min_c = std::min(min_c, lat_cycles[i]);
      max_c = std::max(max_c, lat_cycles[i]);
    }

    const int steady_n = (int)lat_cycles.size() - 1;
    double mean_c = sum / steady_n;

    double var = 0.0;
    for (size_t i = 1; i < lat_cycles.size(); ++i) {
      double d = (double)lat_cycles[i] - mean_c;
      var += d * d;
    }
    // Unbiased estimate if >2 samples
    if (steady_n > 1) var /= (steady_n - 1);
    double sd_c = std::sqrt(var);

    double mean_ns = (1e9 * mean_c) / AIE_clock_Hz;
    double sd_ns   = (1e9 * sd_c) / AIE_clock_Hz;
    double min_ns  = (1e9 * (double)min_c) / AIE_clock_Hz;
    double max_ns  = (1e9 * (double)max_c) / AIE_clock_Hz;

    std::printf("\n-------- Steady-state latency (discard first) --------\n");
    std::printf("Mean : %.1f cycles (%.3f ns)\n", mean_c, mean_ns);
    std::printf("Stdev: %.1f cycles (%.3f ns)\n", sd_c, sd_ns);
    std::printf("Min  : %lld cycles (%.3f ns)\n", min_c, min_ns);
    std::printf("Max  : %lld cycles (%.3f ns)\n\n", max_c, max_ns);
    std::printf("\n\n\n--------GRAPH LATENCY    (First in  -> First out) : %lld cycles, %.1f ns\n\n\n", (long long)mean_c, mean_ns);
  } else {
    std::printf("Not enough iterations to compute steady-state stats.\n");
  }

  return 0;
}
