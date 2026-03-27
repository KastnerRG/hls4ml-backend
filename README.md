# Test repo for developing HLS4ML AIE Backend

Actual development is being done in [KastnerRG/hls4ml](https://github.com/KastnerRG/hls4ml) in the branch [aie_backend/dev](https://github.com/KastnerRG/hls4ml/tree/aie_backend/dev). That branch is included here as `git submodule`

```
git pull --recurse-submodules
conda env create -f environment.yml
conda activate hls4ml-tutorial
python quickstart.py
```

## To run the golden model of NN end-to-end

```
cd golden
python run_workload.py
```

## To run AIE+PL

```bash
source /tools/Xilinx/Vivado/2025.2/Vitis/settings64.sh
cd aie_pl

# edit gen.py to set layers

make sim

# AIE : 570 cycles  (456.0 ns @ 1.25 GHz)  [aiesimulator]
# PL  : 512 cycles  (1638.4 ns @ 312.5 MHz)  [est.]
# Total (analytical): 2094.4 ns  →  3.82 M samples/s

make run_emu

# End-to-end (hw_emu): 1.92224e+09 ns
# PASS
```