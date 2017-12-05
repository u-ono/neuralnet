import numpy as np
from functions import *
import numba

class Affine:

    def __init__(self, W):
        self.W = W
        self.pz = None
        self.dW = None

    def forward(self, pz):
        batch_size = pz.shape[0]

        # calculate z
        biased_pz = np.hstack((np.ones((batch_size,1)),pz))
        self.pz = biased_pz
        z = np.dot(biased_pz, self.W.T)
        
        return z

    def backward(self, err):
        delta = err

        # calculate dW
        batch_size = self.pz.shape[0]
        pre_size = self.pz.shape[1]
        cur_size = delta.shape[1]
        dW = np.zeros((cur_size, pre_size, batch_size))
        for i in range(batch_size):
            dW[:,:,i] = np.dot(delta[i, np.newaxis].T, self.pz[np.newaxis, i])
        self.dW = np.sum(dW, axis=2)

        # calculate delta of the former layer
        delta = np.dot(err, self.W)

        return delta[:, 1:]
        
class Sigmoid:

    def __init__(self, width, a):
        self.z = None
        self.a = a
        self.width = width

    def forward(self, u):
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
        self.z = None
        self.u = None
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
        self.z = None
        self.width = width
        
    def forward(self, u):
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
        self.z = None
        self.width = width

    def forward(self, u):
        self.z = u
        return u

    def backward(self, delta):
        err = delta
        return err

    def dZ(self):
        return np.identity(self.width)

class CrossEntropyError:

    def __init__(self):
        self.out = None
        self.t = None
        self.z = None

    def forward(self, t, z):
        self.t = t
        self.z = z
        out = cross_entropy_error(t, z)
        self.out = out
        return out

    def backward(self, err):
        batch_size = self.t.shape[0]
        delta = -self.t / self.z / batch_size
        return delta

class MeanSquaredError:

    def __init__(self):
        self.out = None
        self.t = None
        self.z = None

    def forward(self, t, z):
        self.t = t
        self.z = z
        batch_size = self.t.shape[0]
        out = mean_squared_error(t, z) / batch_size
        self.out = out
        return out

    def backward(self, err):
        batch_size = self.t.shape[0]
        delta = (self.z - self.t) / batch_size
        return delta
