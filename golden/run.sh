#!/bin/bash

v++ -c --mode aie \
  --include $XILINX_VITIS/aietools/include \
  --include ./aie \
  --include ./model \
  --platform $XILINX_VITIS/base_platforms/xilinx_vck190_base_202410_1/xilinx_vck190_base_202410_1.xpfm \
  --work_dir ./Work \
  --target hw \
  --aie.heapsize=16384 \
  --aie.stacksize=16384 \
  --aie.xlopt=1 \
  aie/graph.cpp
  # --aie.Xxloptstr="-annotate-pragma" \

aiesimulator \
  --pkg-dir=./Work \
  --profile \
  --dump-vcd=aiesim