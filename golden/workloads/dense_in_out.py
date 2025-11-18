import numpy as np
from framework import *

def get_output(batch, inputs, outputs, input_plios, output_plios, dtype, **kwargs):

    x0 = np.random.randint(0, 128, size=(batch, inputs), dtype=TY_DICT[dtype]['np'])

    extra_kwargs = dict(kwargs)
    extra_kwargs.pop('input_plios', None)
    extra_kwargs.pop('output_plios', None)

    model = Sequential(dtype=dtype,
                       input_plios=input_plios,
                       output_plios=output_plios,
                       **extra_kwargs)

    first_kwargs = dict(extra_kwargs)
    first_kwargs['input_plios'] = input_plios
    first_kwargs['cascade_out'] = True
    first_shift = first_kwargs.get('shift', 5)
    first_relu = first_kwargs.get('relu', True)
    m_tile = first_kwargs.get('m_tile', 2)
    n_tile = first_kwargs.get('n_tile', 8)
    model.add(Dense(N=64, shift=first_shift, relu=first_relu, dtype=dtype, **first_kwargs))

    second_kwargs = dict(extra_kwargs)
    second_kwargs['cascade_in'] = True
    second_kwargs['input_shift'] = first_shift
    second_kwargs['input_relu'] = first_relu
    second_kwargs['input_m_tile'] = m_tile
    second_kwargs['input_n_tile'] = n_tile
    if input_plios == 1:
        second_kwargs['cascade_out'] = True
    else:
        second_kwargs['cascade_out'] = False
        second_kwargs['output_plios'] = output_plios
    model.add(Dense(N=outputs, shift=5, relu=True, dtype=dtype, **second_kwargs))

    if output_plios == 1:
        quant_kwargs = dict(extra_kwargs)
        quant_kwargs['m_tile'] = second_kwargs.get('m_tile', 2)
        quant_kwargs['n_tile'] = second_kwargs.get('n_tile', 8)
        model.add(CascadeToStream(shift=5, relu=True, dtype=dtype, **quant_kwargs))

    y_ref = model.build_and_emit(x0)
    return y_ref
