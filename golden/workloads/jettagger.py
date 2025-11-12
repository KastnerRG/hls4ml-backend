import numpy as np
import subprocess
from framework import *

def get_output(batch, dtype, **kwargs):

    inputs = 16
    x0 = np.random.randint(0, 128, size=(batch,inputs), dtype=TY_DICT[dtype]['np'])
    model = Sequential(dtype=dtype, **kwargs)

    model.add(Dense(N=64, shift=5, relu=True, dtype=dtype, **kwargs))
    model.add(Dense(N=32, shift=5, relu=True, dtype=dtype, **kwargs))
    model.add(Dense(N=32, shift=5, relu=True, dtype=dtype, **kwargs))
    model.add(Dense(N=16, shift=5, relu=True, dtype=dtype, **kwargs))

    y_ref = model.build_and_emit(x0)
    subprocess.run(["./run.sh"], check=True)
    return y_ref
