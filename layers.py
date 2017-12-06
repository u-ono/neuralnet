import numpy as np
from functions import *
import numba

class Sigmoid:

    def __init__(self, width, a):
        self.u = None
        self.z = None
        self.a = a
        self.width = width

    def forward(self, u):
        self.u = u
        z = sigmoid(u, self.a)
        self.z = z
        return z
        
    def backward(self, u):
        grad = sigmoid_diff(u, self.a)
        return grad
        
    @numba.jit
    def dZ(self):
        dZ = np.zeros((self.width, self.width))
        for k in range(self.width):
            dZ[k, k] = self.a * self.z[k] * (1 - self.z[k])
        return dZ

class ReLU:

    def __init__(self, width):
        self.u = None
        self.z = None
        self.width = width

    def forward(self, u):
        self.u = u
        z = relu(u)
        self.z = z
        return z

    def backward(self, u):
        grad = relu_diff(u)
        return grad

    def dZ(self):
        if self.u > 0:
            dZ = np.identity(self.width)
        else:
            dZ = np.zeros((self.width, self.width))
        return dZ

class Softmax:

    def __init__(self, width):
        self.u = None
        self.z = None
        self.width = width
        
    def forward(self, u):
        self.u = u
        z = softmax(u)
        self.z = z
        return z

    def backward(self, delta):
        softmax_diff = self.z * (1 - self.z)
        err = delta * softmax_diff
        return err

    @numba.jit
    def dZ(self):
        dZ = np.zeros((self.width, self.width))
        for k in range(self.width):
            for i in range(self.width):
                dZ[k, i] = - self.z[i] * self.z[k]
        for k in range(self.width):
            dZ[k, k] = self.z[k] * (1 - self.z[k])
        return dZ 

class Identity:

    def __init__(self, width):
        self.u = None
        self.z = None
        self.width = width

    def forward(self, u):
        self.u = u
        self.z = u
        return u

    def backward(self, delta):
        err = delta
        return err

    def dZ(self):
        return np.identity(self.width)

class CrossEntropyError:

    def __init__(self):
        self.t = None
        self.z = None
        self.name = 'cross_entropy'
        self.dE = None

    def forward(self, z, t):
        self.t = t
        self.z = z
        out = cross_entropy_error(t, z)
        self.dE = z / t
        return out

class MeanSquaredError:

    def __init__(self):
        self.t = None
        self.z = None
        self.name = 'mean_squared'
        self.dE = None

    def forward(self, z, t):
        self.t = t
        self.z = z
        out = mean_squared_error(t, z)
        self.dE = z - t
        return out
