import numpy as np
from framework import *

def get_output(batch, inputs, outputs, dtype, **kwargs):
    assert inputs == 128, "dense_in_out assumes 128 input features"
    assert outputs == 256, "dense_in_out assumes 256 output features"

    requested_input_plios = kwargs.get("input_plios", 8)
    requested_output_plios = kwargs.get("output_plios", 16)
    assert requested_input_plios == 8, "First layer requires 8 input PLIOs"
    assert requested_output_plios == 16, "Second layer requires 16 output PLIOs"

    x0 = np.random.randint(0, 128, size=(batch, inputs), dtype=TY_DICT[dtype]['np'])

    extra_kwargs = dict(kwargs)
    extra_kwargs.pop('input_plios', None)
    extra_kwargs.pop('output_plios', None)

    model = Sequential(dtype=dtype,
                       input_plios=requested_input_plios,
                       output_plios=requested_output_plios,
                       **extra_kwargs)

    first_kwargs = dict(extra_kwargs)
    first_kwargs['input_plios'] = requested_input_plios
    model.add(Dense(N=16, shift=5, relu=True, dtype=dtype, **first_kwargs))

    second_kwargs = dict(extra_kwargs)
    second_kwargs['output_plios'] = requested_output_plios
    model.add(Dense(N=outputs, shift=5, relu=True, dtype=dtype, **second_kwargs))

    y_ref = model.build_and_emit(x0)
    return y_ref
