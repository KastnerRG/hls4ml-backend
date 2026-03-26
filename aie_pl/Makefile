SHELL := /bin/bash

CONDA_LIB ?= $(HOME)/install/miniconda/envs/aie_pl/lib
export LD_LIBRARY_PATH := $(CONDA_LIB)$(if $(LD_LIBRARY_PATH),:$(LD_LIBRARY_PATH))

VITIS_HOME ?= $(strip $(shell dirname "$$(dirname "$$(which vitis 2>/dev/null)")" 2>/dev/null))
PLATFORM   ?= $(VITIS_HOME)/base_platforms/xilinx_vek280_base_202520_1/xilinx_vek280_base_202520_1.xpfm

AIE_SRC    := aie/graph.cpp
AIE_INC    := --include=aie --include=aie/weights
AIE_FLAGS  := $(AIE_INC) --platform=$(PLATFORM) \
              --aie.xlopt=1 --aie.Xmapper=BufferOptLevel8 \
              --aie.heapsize=16420 --aie.stacksize=4096

.PHONY: compile sim x86compile x86sim clean

compile: Work/.done

Work/.done: $(AIE_SRC) aie/parameters.h $(wildcard aie/weights/*.h) \
            aie/dense.h aie/dense.cpp aie/dense_graph.h
	set -o pipefail; v++ --compile --mode aie --target hw $(AIE_FLAGS) \
	    --aie.workdir=Work $(AIE_SRC) 2>&1 | tee aie_compile.log
	@touch Work/.done

sim: Work/.done
	set -o pipefail; aiesimulator --pkg-dir=Work --profile 2>&1 | tee aie_sim.log

x86compile: Work_x86/.done

Work_x86/.done: $(AIE_SRC) aie/parameters.h $(wildcard aie/weights/*.h) \
                aie/dense.h aie/dense.cpp aie/dense_graph.h
	set -o pipefail; v++ --compile --mode aie --target x86sim $(AIE_FLAGS) \
	    --aie.workdir=Work_x86 $(AIE_SRC) 2>&1 | tee x86_compile.log
	@touch Work_x86/.done

x86sim: Work_x86/.done
	set -o pipefail; x86simulator --pkg-dir=Work_x86 2>&1 | tee x86_sim.log

clean:
	rm -rf Work Work_x86 libadf.a libadf_x86.a
	rm -rf aiesimulator_output x86simulator_output
	rm -rf *.log .Xil .AIE_SIM_CMD_LINE_OPTIONS ISS_RPC_SERVER_PORT
	rm -rf aiesim.vcd Map_Report.csv sol.db pl_sample_counts plio_throughput_info.json
	rm -rf vitis_analyzer* temp function_wdb_dir
