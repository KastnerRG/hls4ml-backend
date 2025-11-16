import numpy as np
from framework import *

def get_output(batch, inputs, outputs, dtype, plio_out=4, **kwargs):

    x0 = np.random.randint(0, 128, size=(batch, inputs), dtype=TY_DICT[dtype]['np'])
    extra_kwargs = dict(kwargs)
    extra_kwargs.pop('output_plios', None)
    model = Sequential(dtype=dtype, output_plios=plio_out, **extra_kwargs)

    model.add(Dense(N=outputs, shift=5, relu=True, dtype=dtype, output_plios=plio_out, **extra_kwargs))

    y_ref = model.build_and_emit(x0)
    return y_ref
