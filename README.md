# Test repo for developing HLS4ML AIE Backend

Actual development is being done in [KastnerRG/hls4ml](https://github.com/KastnerRG/hls4ml) in the branch [aie_backend/dev](https://github.com/KastnerRG/hls4ml/tree/aie_backend/dev). That branch is included here as `git submodule`

```
git pull --recurse-submodules

cd hls4ml
conda install -c conda-forge tensorflow=2.8
pip install .[qkeras]

cd ..
python quickstart.py
```