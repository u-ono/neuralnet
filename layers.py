import numpy as np
from functions import *

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

    def __init__(self, a):
        self.a = a

    def forward(self, u):
        z = sigmoid(u, self.a)
        return z
        
    def backward(self, u):
        grad = sigmoid_diff(u, self.a)
        return grad

class ReLU:

    def __init__(self):
        self.z = None

    def forward(self, u):
        z = relu(u)
        return z

    def backward(self, u):
        grad = relu_diff(u)
        return grad

class Softmax:

    def __init__(self):
        self.z = None
        
    def forward(self, u):
        z = softmax(u)
        self.z = z
        return z

    def backward(self, delta):
        softmax_diff = self.z * (1 - self.z)
        err = delta * softmax_diff
        return err

class Identity:

    def __init__(self):
        self.z = None

    def forward(self, u):
        self.z = u
        return u

    def backward(self, delta):
        err = delta
        return err

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
