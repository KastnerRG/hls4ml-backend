import numpy as np
import subprocess
from framework import *

def get_output(batch, inputs, outputs, dtype, dataflow, iterations, m_tile, k_tile, n_tile):

    inputs = 16
    x0 = np.random.randint(0, 128, size=(batch,inputs), dtype=TY_DICT[dtype]['np'])
    model = Sequential(iterations=iterations, dtype=dtype, dataflow=dataflow)

    model.add(Dense(N=64, shift=5, relu=True, m_tile=m_tile, k_tile=k_tile, n_tile=n_tile, dtype=dtype, dataflow=dataflow))
    model.add(Dense(N=32, shift=5, relu=True, m_tile=m_tile, k_tile=k_tile, n_tile=n_tile, dtype=dtype, dataflow=dataflow))
    model.add(Dense(N=32, shift=5, relu=True, m_tile=m_tile, k_tile=k_tile, n_tile=n_tile, dtype=dtype, dataflow=dataflow))
    model.add(Dense(N=16, shift=5, relu=True, m_tile=m_tile, k_tile=k_tile, n_tile=n_tile, dtype=dtype, dataflow=dataflow))

    y_ref = model.build_and_emit(x0)
    subprocess.run(["./run.sh"], check=True)
    return y_ref
