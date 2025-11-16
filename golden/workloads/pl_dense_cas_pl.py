import numpy as np
from framework import *

def get_output(batch, inputs, outputs, dtype, **kwargs):
    assert inputs == 128 and outputs == 16, "pl_dense_cas_pl fixed at 128 inputs, 16 outputs"

    m_tile = kwargs.get('m_tile', 2)
    k_tile = kwargs.get('k_tile', 8)
    n_tile = kwargs.get('n_tile', 8)
    iterations = kwargs.get('iterations', 1)
    free = kwargs.get('free', False)

    x0 = np.random.randint(0, 128, size=(batch, inputs), dtype=TY_DICT[dtype]['np'])

    model = Sequential(dtype=dtype, dataflow='stream', iterations=iterations, free=free, input_plios=8, output_plios=1)

    model.add(Dense(N=16, shift=5, relu=True, dtype=dtype, dataflow='stream',
                    free=free, m_tile=m_tile, k_tile=k_tile, n_tile=n_tile,
                    input_plios=8, output_plios=1, cascade_out=True))

    model.add(CascadeToStream(shift=5, relu=True, m_tile=m_tile, n_tile=n_tile,
                              k_tile=k_tile, dtype=dtype, free=free))

    y_ref = model.build_and_emit(x0)
    return y_ref
