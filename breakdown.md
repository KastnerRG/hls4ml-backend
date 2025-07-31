# Breakdown of HLS4ML Backend Generation

1. User's source file with model definition: `quickstart.py`
2. HLS model definition: `firmware/${PROJECT}.cpp`
3. HLS Python Bridge: `firmware/${PROJECT}_bridge.cpp`
4. Weights (as .h files): `firmware/weights/*.h`
5. Actual layers: `firmware/nnet_utils/` copied from `hls4ml/templates/vivado/nnet_utils`
6. C++ compile commands: `hls4ml/templates/vivado/build_lib.sh`

```
hls_model = hls4ml.converters.convert_from_keras_model(backend='Vitis')
  [converters/__init__.py] convert_from_keras_model(): keras_to_hls(config)
    [converters/keras_to_hls.py] keras_to_hls(): ModelGraph(config, layer_list, input_layers, output_layers)
      [model/graph.py] ModelGraph.__init__

hls_model.compile()
  [model/graph.py] ModelGraph.compile()
    [model/graph.py] ModelGraph.write(): self.config.backend.write(self)
      

    [model/graph.py] ModelGraph._compile(): lib_name = self.config.backend.compile(self)
      [backends/fpga/fpga_backend.py] FPGABackend.compile: ret_val = subprocess.run('build_lib.sh')
        [templates/vivado/build_lib.sh]: C++ compile
```
  