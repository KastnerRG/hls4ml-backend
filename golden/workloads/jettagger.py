import numpy as np
from framework import *

def get_output(batch, dtype, **kwargs):

    inputs = 128
    base_kwargs = dict(kwargs)
    base_kwargs.pop('input_plios', None)
    base_kwargs.pop('output_plios', None)
    x0 = np.random.randint(0, 128, size=(batch, inputs), dtype=TY_DICT[dtype]['np'])

    base_kwargs.setdefault('m_tile', 2)
    base_kwargs.setdefault('k_tile', 8)
    base_kwargs.setdefault('n_tile', 8)

    model = Sequential(dtype=dtype, input_plios=8, output_plios=1, **base_kwargs)

    first_kwargs = dict(base_kwargs)
    first_kwargs['input_plios'] = 8
    first_kwargs['cascade_out'] = True
    first_shift = first_kwargs.get('shift', 5)
    first_relu = first_kwargs.get('relu', True)
    m_tile = first_kwargs['m_tile']
    k_tile = first_kwargs['k_tile']
    n_tile = first_kwargs['n_tile']
    model.add(Dense(N=64, shift=first_shift, relu=first_relu, dtype=dtype, **first_kwargs))

    second_kwargs = dict(base_kwargs)
    second_kwargs['cascade_in'] = True
    second_kwargs['input_shift'] = first_shift
    second_kwargs['input_relu'] = first_relu
    second_kwargs['input_m_tile'] = m_tile
    second_kwargs['input_n_tile'] = k_tile
    second_kwargs['input_plios'] = 4
    second_kwargs['cascade_out'] = True
    second_shift = second_kwargs.get('shift', 5)
    second_relu = second_kwargs.get('relu', True)
    model.add(Dense(N=32, shift=second_shift, relu=second_relu, dtype=dtype, **second_kwargs))

    third_kwargs = dict(base_kwargs)
    third_kwargs['cascade_in'] = True
    third_kwargs['input_shift'] = second_shift
    third_kwargs['input_relu'] = second_relu
    third_kwargs['input_m_tile'] = m_tile
    third_kwargs['input_n_tile'] = k_tile
    third_kwargs['input_plios'] = 2
    third_kwargs['cascade_out'] = True
    third_shift = third_kwargs.get('shift', 5)
    third_relu = third_kwargs.get('relu', True)
    model.add(Dense(N=32, shift=third_shift, relu=third_relu, dtype=dtype, **third_kwargs))

    fourth_kwargs = dict(base_kwargs)
    fourth_kwargs['cascade_in'] = True
    fourth_kwargs['input_shift'] = third_shift
    fourth_kwargs['input_relu'] = third_relu
    fourth_kwargs['input_m_tile'] = m_tile
    fourth_kwargs['input_n_tile'] = k_tile
    fourth_kwargs['input_plios'] = 1
    fourth_kwargs['cascade_out'] = False
    fourth_kwargs['output_plios'] = 1
    model.add(Dense(N=16, shift=5, relu=True, dtype=dtype, **fourth_kwargs))

    y_ref = model.build_and_emit(x0)
    return y_ref
