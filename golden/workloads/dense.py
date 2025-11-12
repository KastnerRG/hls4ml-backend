import numpy as np
import subprocess
from framework import *

def get_output(batch, inputs, outputs, dtype, **kwargs):

    x0 = np.random.randint(0, 128, size=(batch, inputs), dtype=TY_DICT[dtype]['np'])
    model = Sequential(dtype=dtype, **kwargs)

    model.add(Dense(N=outputs, shift=5, relu=True, dtype=dtype, **kwargs))

    y_ref = model.build_and_emit(x0)
    subprocess.run(["./run.sh"], check=True)
    return y_ref