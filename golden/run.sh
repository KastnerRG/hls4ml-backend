mkdir -p data
python3 golden.py
v++ -c --mode aie --include $XILINX_VITIS/aietools/include --include "./aie" --include "./data" --include "./aie/kernels" --include "./" --aie.xlopt=0 --platform $XILINX_VITIS/base_platforms/xilinx_vck190_base_202410_1/xilinx_vck190_base_202410_1.xpfm --work_dir ./Work --target hw aie/graph.cpp
aiesimulator --profile --dump-vcd=tutorial --pkg-dir=./Work

grep -v '^T' "aiesimulator_output/data/matC0.txt" > "data/matC0_sim.txt"
diff -w "data/matC0_sim.txt" "data/matC0.txt" > /dev/null  && echo "\n\n Success: Outputs match\n\n" || echo "\n\nError: Output does not match\n\n"