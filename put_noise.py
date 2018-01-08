#!/usr/bin/env python

import numpy as np
from numba import jit

@jit('f4[:, :](f4[:, :],i8)')
def put_noise(data, d):
    ret = np.zeros_like(data)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if np.random.randint(100) < d:
                ret[i, j] = np.random.rand()
            else:
                ret[i, j] = data[i, j]
    return ret
