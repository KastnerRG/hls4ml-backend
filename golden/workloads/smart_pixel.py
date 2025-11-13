import numpy as np
from framework import *

def get_output(batch, inputs, outputs, dtype, **kwargs):
    x0 = np.random.randint(0, 128, size=(13, 21, 20), dtype=TY_DICT[dtype]['np'])
    model = Sequential(dtype=dtype, **kwargs)

    model.add(
        Conv2d(
            XH=13,
            XW=21,
            CI=20,
            CO=5,
            KH=5,
            KW=3,
            stride=(1, 1),
            padding=(1, 0),
            shift=5,
            relu=True,
            dtype=dtype,
            **kwargs,
        )
    )

    model.add(
        Conv2d(
            XH=11,
            XW=19,
            CI=5,
            CO=5,
            KH=5,
            KW=1,
            stride=(1, 1),
            padding=(2, 0),
            shift=5,
            relu=True,
            dtype=dtype,
            **kwargs,
        )
    )

    y_ref = model.build_and_emit(x0)
    return y_ref
