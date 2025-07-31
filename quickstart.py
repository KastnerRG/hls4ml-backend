import sys
sys.path.append("hls4ml/")
import hls4ml
import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Dense, Activation

model = tf.keras.models.Sequential()
model.add(Dense(64, input_shape=(16,), name='Dense', kernel_initializer='lecun_uniform', kernel_regularizer=None))
model.add(Activation(activation='relu', name='Activation'))
model.add(Dense(32, name='Dense2', kernel_initializer='lecun_uniform', kernel_regularizer=None))
model.add(Activation(activation='relu', name='Activation2'))


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
