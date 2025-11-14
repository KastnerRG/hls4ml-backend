import numpy as np
from framework import *

def get_output(batch, inputs, outputs, dtype, **kwargs):
    x0 = np.random.randint(0, 128, size=(batch, 13, 21, 20), dtype=TY_DICT[dtype]['np'])
    model = Sequential(dtype=dtype, batch=batch, **kwargs)

    conv0 = ConvAsDense(
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
        batch=batch,
        **kwargs,
    )
    model.add(conv0)

    conv1 = ConvAsDense(
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
        batch=batch,
        **kwargs,
    )
    model.add(conv1)

    flatten_dense = FlattenDense(
        input_size=3*6*5,
        output_size=16,
        shift=5,
        relu=True,
        dtype=dtype,
        batch=batch,
        **kwargs,
    )

    model.add(
        ConvPoolFlattenPrep(
            conv=conv1,
            pool_kernel=(3, 3),
            pool_stride=(3, 3),
            pool_padding=(0, 0),
            flatten_layer=flatten_dense,
        )
    )

    model.add(flatten_dense)

    model.add(
        Dense(
            N=16,
            shift=5,
            relu=True,
            dtype=dtype,
            **kwargs,
        )
    )

    model.add(
        Dense(
            N=16,
            shift=5,
            relu=True,
            dtype=dtype,
            **kwargs,
        )
    )

    y_ref = model.build_and_emit(x0)
    return y_ref
