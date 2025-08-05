import os
import sys
sys.path.append("hls4ml/")
import tensorflow as tf
from dotenv import load_dotenv
import numpy as np
from tensorflow.keras.layers import Dense, Activation
from qkeras.qlayers import QDense, QActivation
from qkeras.quantizers import quantized_bits, quantized_relu
import hls4ml


load_dotenv(dotenv_path=".env")
os.environ['PATH'] += os.pathsep + '/tools/Xilinx/Vitis/2024.1/bin' + os.pathsep + '/tools/Xilinx/Vivado/2024.1/bin' + os.pathsep + '/opt/xilinx/xrt/bin'

model = tf.keras.models.Sequential()
model.add(
   QDense(
      64, 
      input_shape=(16,), 
      name='Dense',
      use_bias=False,
      kernel_quantizer=quantized_bits(8, 0, alpha=1),
      kernel_initializer='lecun_uniform', 
      kernel_regularizer=None
   )
)
model.add(QActivation(activation=quantized_relu(8), name='Activation'))
model.add(
   QDense(
      32, 
      name='Dense2', 
      use_bias=False,
      kernel_quantizer=quantized_bits(8, 0, alpha=1),
      kernel_initializer='lecun_uniform', 
      kernel_regularizer=None
   )
)
model.add(QActivation(activation=quantized_relu(8), name='Activation2'))


config = hls4ml.utils.config_from_keras_model(model)
print(config)

hls_model = hls4ml.converters.convert_from_keras_model(
   model=model,
   hls_config=config,
   backend='Vitis'
)
hls_model.compile()

X_input = np.random.rand(100, 16)
hls_prediction = hls_model.predict(X_input)

hls_model.build()

hls4ml.report.read_vivado_report('my-hls-test')
