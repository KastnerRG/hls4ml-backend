import numpy as np
from framework import *

def get_output(batch, inputs, outputs, dtype, **kwargs):
    assert inputs == 128, "dense_in_out assumes 128 input features"
    assert outputs == 256, "dense_in_out assumes 256 output features"

    requested_input_plios = kwargs.get("input_plios", 8)
    requested_output_plios = kwargs.get("output_plios", 1)
    assert requested_input_plios == 8, "First layer requires 8 input PLIOs"

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
    first_kwargs['cascade_out'] = True
    first_shift = first_kwargs.get('shift', 5)
    first_relu = first_kwargs.get('relu', True)
    m_tile = first_kwargs.get('m_tile', 2)
    n_tile = first_kwargs.get('n_tile', 8)
    model.add(Dense(N=16, shift=first_shift, relu=first_relu, dtype=dtype, **first_kwargs))

    second_kwargs = dict(extra_kwargs)
    second_kwargs['cascade_in'] = True
    second_kwargs['input_shift'] = first_shift
    second_kwargs['input_relu'] = first_relu
    second_kwargs['input_m_tile'] = m_tile
    second_kwargs['input_n_tile'] = n_tile
    second_kwargs['cascade_out'] = True
    model.add(Dense(N=outputs, shift=5, relu=True, dtype=dtype, **second_kwargs))

    quant_kwargs = dict(extra_kwargs)
    quant_kwargs['m_tile'] = second_kwargs.get('m_tile', 2)
    quant_kwargs['n_tile'] = second_kwargs.get('n_tile', 8)
    model.add(CascadeToStream(shift=5, relu=True, dtype=dtype, **quant_kwargs))

    y_ref = model.build_and_emit(x0)
    return y_ref
