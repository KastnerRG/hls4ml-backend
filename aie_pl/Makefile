SHELL := /bin/bash

# ── Environment ───────────────────────────────────────────────────────────────
CONDA_LIB ?= $(HOME)/install/miniconda/envs/aie_pl/lib
export LD_LIBRARY_PATH := $(CONDA_LIB)$(if $(LD_LIBRARY_PATH),:$(LD_LIBRARY_PATH))

VITIS_HOME ?= /tools/Xilinx/Vivado/2025.2/Vitis
PLATFORM   ?= $(VITIS_HOME)/base_platforms/xilinx_vek280_base_202520_1/xilinx_vek280_base_202520_1.xpfm

export XILINX_VITIS          := $(VITIS_HOME)
export XILINX_HLS            := $(VITIS_HOME)
export XILINX_VIVADO         := $(VITIS_HOME)/../Vivado
export XILINXD_LICENSE_FILE  ?= 2100@cselm2.ucsd.edu
export PATH                  := $(VITIS_HOME)/bin:$(VITIS_HOME)/aietools/bin:$(XILINX_VIVADO)/bin:$(PATH)

# ── Versal common image ───────────────────────────────────────────────────────
# sdk.sh was installed to VERSAL_SDK; rootfs.ext4 and Image live there too.
VERSAL_SDK    ?= $(HOME)/install/Xilinx/versal-common-2025.2
VERSAL_ROOTFS ?= $(VERSAL_SDK)

SDKTARGETSYSROOT ?= $(VERSAL_SDK)/sysroots/cortexa72-cortexa53-amd-linux
ROOTFS           ?= $(VERSAL_ROOTFS)/rootfs.ext4
IMAGE            ?= $(VERSAL_ROOTFS)/Image

TARGET ?= hw_emu

# ── aarch64 cross-compiler ────────────────────────────────────────────────────
VERSAL_CXX_BIN   := $(VERSAL_SDK)/sysroots/x86_64-petalinux-linux/usr/bin/aarch64-amd-linux/aarch64-amd-linux-g++
VERSAL_CXX_FLAGS := -mcpu=cortex-a72.cortex-a53+crc -mbranch-protection=standard \
                    -fstack-protector-strong -O2 -D_FORTIFY_SOURCE=2 \
                    -Wformat -Wformat-security -Werror=format-security \
                    --sysroot=$(SDKTARGETSYSROOT)

ifeq ($(origin CXX),default)
HOST_CXX := $(VERSAL_CXX_BIN) $(VERSAL_CXX_FLAGS)
else
HOST_CXX := $(CXX)
endif

# ── Knobs (override on command line: make BATCH=16 sim) ───────────────────────
BATCH        ?= 8
SHIFT        ?= 0
N_ITER       ?= 1
CAS_LENGTH   ?= 1
CAS_NUM      ?= 1
TILE_M       ?= 4
TILE_K       ?= 8
TILE_N       ?= 8
REUSE_FACTOR ?= 1
SEED         ?= 42

GEN_ARGS := --batch=$(BATCH) --shift=$(SHIFT) --n-iter=$(N_ITER) \
            --cas-length=$(CAS_LENGTH) --cas-num=$(CAS_NUM) \
            --tile-m=$(TILE_M) --tile-k=$(TILE_K) --tile-n=$(TILE_N) \
            --reuse-factor=$(REUSE_FACTOR) --seed=$(SEED)

# Write params.stamp only when knob values change (preserves mtime otherwise).
# Runs at Makefile parse time, before any targets are evaluated.
$(shell echo '$(GEN_ARGS)' | diff -q - params.stamp >/dev/null 2>&1 || \
        echo '$(GEN_ARGS)' > params.stamp)

# Bootstrap: run gen.py now if groups.mk is missing (first invocation after clean).
# This ensures PL_GROUP_XOS is populated before any targets are evaluated.
ifeq ($(wildcard generated/groups.mk),)
$(shell python gen.py $(GEN_ARGS) >/dev/null 2>&1)
endif
-include generated/groups.mk

JOBS := $(shell nproc)
MAKEFLAGS += -j$(JOBS)

# ── AIE compile flags ─────────────────────────────────────────────────────────
AIE_FLAGS := --include=aie --include=aie/weights --platform=$(PLATFORM) \
             --aie.xlopt=1 --aie.Xmapper=BufferOptLevel8 \
             --aie.heapsize=16420 --aie.stacksize=4096

.PHONY: sim x86sim kernels hw_emu run_emu gen check clean

# ── Default target ────────────────────────────────────────────────────────────
sim: check

# ── Code generation ───────────────────────────────────────────────────────────
# gen.py is fast and uses write_if_changed internally — always run it, only
# files whose content changes get a new mtime, gating the heavy steps below.
aie/graph.cpp aie/parameters.h sw/host.cpp sw/data.h sw/embedded_exec.sh system.cfg \
pl_kernels/mm2s.cfg pl_kernels/s2mm.cfg generated/groups.mk &: gen.py params.stamp
	python gen.py $(GEN_ARGS)

# Pattern rule so Make knows pl_group*.cfg come from gen.py.
pl_kernels/pl_group%.cfg: gen.py params.stamp
	python gen.py $(GEN_ARGS)

gen: aie/graph.cpp

# ── AIE compile ───────────────────────────────────────────────────────────────
Work/.done: aie/graph.cpp aie/parameters.h aie/dense.h aie/dense.cpp aie/dense_graph.h
	set -o pipefail; v++ --compile --mode aie --target hw $(AIE_FLAGS) \
	    --aie.workdir=Work aie/graph.cpp 2>&1 | tee aie_compile.log
	@touch Work/.done

Work_x86/.done: aie/graph.cpp aie/parameters.h aie/dense.h aie/dense.cpp aie/dense_graph.h
	set -o pipefail; v++ --compile --mode aie --target x86sim $(AIE_FLAGS) \
	    --aie.workdir=Work_x86 aie/graph.cpp 2>&1 | tee x86_compile.log
	@touch Work_x86/.done

# ── Simulation ────────────────────────────────────────────────────────────────
aiesimulator_output/.done: Work/.done
	set -o pipefail; aiesimulator --pkg-dir=Work --profile 2>&1 | tee aie_sim.log
	@touch $@

x86simulator_output/.done: Work_x86/.done
	set -o pipefail; x86simulator --pkg-dir=Work_x86 2>&1 | tee x86_sim.log
	@touch $@

check: aiesimulator_output/.done
	python check.py

x86sim: x86simulator_output/.done
	python check.py

# ── hw_emu: PL kernels ────────────────────────────────────────────────────────
pl_kernels/mm2s.xo: pl/mm2s.cpp pl_kernels/mm2s.cfg
	set -o pipefail; v++ -c --mode hls --platform $(PLATFORM) \
	    --config pl_kernels/mm2s.cfg 2>&1 | tee $@.log

pl_kernels/s2mm.xo: pl/s2mm.cpp pl_kernels/s2mm.cfg
	set -o pipefail; v++ -c --mode hls --platform $(PLATFORM) \
	    --config pl_kernels/s2mm.cfg 2>&1 | tee $@.log

pl_kernels/pl_group%.xo: pl/pl_group%.cpp pl_kernels/pl_group%.cfg pl/dense.h
	set -o pipefail; v++ -c --mode hls --platform $(PLATFORM) \
	    --config pl_kernels/pl_group$*.cfg 2>&1 | tee $@.log

kernels: pl_kernels/mm2s.xo pl_kernels/s2mm.xo $(PL_GROUP_XOS)

# ── hw_emu: link ──────────────────────────────────────────────────────────────
libadf.a: Work/.done
	@test -f libadf.a || (echo "ERROR: libadf.a missing after AIE compile" && exit 1)

design.xsa: libadf.a pl_kernels/mm2s.xo pl_kernels/s2mm.xo $(PL_GROUP_XOS) system.cfg
	set -o pipefail; v++ -l -t $(TARGET) --platform $(PLATFORM) \
	    libadf.a pl_kernels/mm2s.xo pl_kernels/s2mm.xo $(PL_GROUP_XOS) \
	    --config system.cfg --save-temps -g \
	    -o design.xsa 2>&1 | tee link.log

xsa: design.xsa

# ── hw_emu: host ──────────────────────────────────────────────────────────────
sw/host.exe: sw/host.cpp sw/data.h sw/embedded_exec.sh
	@test -d "$(SDKTARGETSYSROOT)" || \
	    (echo "ERROR: SDKTARGETSYSROOT='$(SDKTARGETSYSROOT)' not found." && \
	     echo "  Set VERSAL_SDK=<path where sdk.sh was installed>" && exit 1)
	@test -f "$(ROOTFS)" || \
	    (echo "ERROR: ROOTFS='$(ROOTFS)' not found." && \
	     echo "  Set VERSAL_ROOTFS=<dir containing rootfs.ext4 and Image>" && exit 1)
	set -o pipefail; $(HOST_CXX) -Wall -std=c++17 \
	    -I$(SDKTARGETSYSROOT)/usr/include/xrt \
	    -I$(SDKTARGETSYSROOT)/usr/include \
	    -I$(VITIS_HOME)/aietools/include \
	    -I$(VITIS_HOME)/include \
	    -I sw \
	    sw/host.cpp \
	    -L$(VITIS_HOME)/aietools/lib/aarch64.o \
	    -ladf_api_xrt \
	    -lxrt_coreutil \
	    -L$(SDKTARGETSYSROOT)/usr/lib \
	    -o sw/host.exe \
	    2>&1 | tee host.log

host: sw/host.exe

# ── hw_emu: package & run ─────────────────────────────────────────────────────
sw/launch_hw_emu.sh: design.xsa sw/host.exe
	set -o pipefail; cd sw && v++ -p -t $(TARGET) \
	    -f $(PLATFORM) \
	    --package.rootfs=$(ROOTFS) \
	    --package.image_format=ext4 \
	    --package.boot_mode=sd \
	    --package.kernel_image=$(IMAGE) \
	    --package.defer_aie_run \
	    --package.sd_file embedded_exec.sh \
	    --package.sd_file host.exe \
	    ../design.xsa ../libadf.a \
	    2>&1 | tee ../package.log

package: sw/launch_hw_emu.sh

hw_emu: kernels xsa host package

run_emu: sw/launch_hw_emu.sh
	cd sw && ./launch_hw_emu.sh \
	    -run-app embedded_exec.sh \
	    -add-env AIE_COMPILER_WORKDIR=../Work

# ── Convenience aliases ───────────────────────────────────────────────────────

# ── Clean ─────────────────────────────────────────────────────────────────────
clean:
	rm -rf Work Work_x86 libadf.a libadf_x86.a
	rm -rf aiesimulator_output x86simulator_output
	rm -rf *.log .Xil .AIE_SIM_CMD_LINE_OPTIONS ISS_RPC_SERVER_PORT
	rm -rf aiesim.vcd Map_Report.csv sol.db pl_sample_counts plio_throughput_info.json
	rm -rf vitis_analyzer* temp function_wdb_dir _x analyzer_input *.link_summary
	rm -rf pl_kernels _x pl/pl_group*.cpp pl/pl_weights mm2s/ s2mm/ pl_group*/
	rm -rf design.xsa *.xsa system.cfg
	rm -rf sw/
	rm -rf data/ generated/ params.stamp aie/graph.cpp aie/parameters.h aie/weights/
