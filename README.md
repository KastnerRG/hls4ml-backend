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
# PL  : 127 cycles  (406.4 ns @ 312.5 MHz)  [HLS report]
# Total (analytical): 862.4 ns  →  9.28 M samples/s

make run_emu
```