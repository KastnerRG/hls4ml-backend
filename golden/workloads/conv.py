import numpy as np
from framework import *

def get_output(batch, inputs, outputs, dtype, **kwargs):
    # Force a tiny test kernel regardless of CLI arguments
    XH = min(batch, 4)
    XW = min(batch, 4)
    CI = 4
    CO = 8

    per_plio = TY_DICT[dtype]['per_plio']
    assert (XH * XW * CI) % per_plio == 0, "Input volume must align to PLIO width for streaming conv"
    assert (XH * XW * CO) % per_plio == 0, "Output volume must align to PLIO width for streaming conv"

    x0 = np.random.randint(0, 128, size=(XH, XW, CI), dtype=TY_DICT[dtype]['np'])
    model = Sequential(dtype=dtype, **kwargs)

    model.add(
        Conv2d(
            XH=XH,
            XW=XW,
            CI=CI,
            CO=CO,
            KH=3,
            KW=3,
            stride=(1, 1),
            padding=(1, 1),
            shift=5,
            relu=True,
            dtype=dtype,
            **kwargs,
        )
    )

    y_ref = model.build_and_emit(x0)
    return y_ref
