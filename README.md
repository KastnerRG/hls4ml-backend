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
python golden.py
```