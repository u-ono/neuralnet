import numpy as np
from layers import *

class NeuralNet:

    def __init__(self, form, activ_func, loss_func, std = 0.01):

        self.depth = len(form)-1

        self.W = {}
        self.B = {}
        self.dW = {}
        self.dB = {}

        for l in range(1, self.depth+1):
            # settings of W, B, dW, dB
            name = 'layer{}'.format(l)
            self.W[name] = std * np.random.randn(form[l], form[l-1])
            self.dW[name] = np.zeros((form[l], form[l-1]))
            self.B[name] = std * np.random.randn(form[l])
            self.dB[name] = np.zeros(form[l])

        self.Z = {}
        self.U = {}
        if activ_func == 'sigmoid':
            self.h = Sigmoid(1.0)
        elif activ_func == 'relu':
            self.h = ReLU()

    def forprop(self, x):

        z = x
        self.Z['layer0'] = z
        for l in range(1, self.depth+1):
            name = 'layer{}'.format(l)
            u = np.dot(z, self.W[name].T)
            u += self.B[name]
            self.U[name] = u
            z = self.h.forward(u)
            self.Z[name] = z

        return z
        
    def loss(self, z, t):

        delta = (z - t) * 1
        name = 'layer{}'.format(self.depth)
        pre_name = 'layer{}'.format(self.depth-1)
        self.dW[name] += np.dot(delta[:, np.newaxis], self.Z[pre_name][np.newaxis, :])
        self.dB[name] += delta

        return delta

    def backprop(self, delta):

        for l in range(self.depth-1, 0, -1):
            name = 'layer{}'.format(l)
            post_name = 'layer{}'.format(l+1)
            pre_name = 'layer{}'.format(l-1)
            delta = self.h.backward(self.U[name]) * np.dot(delta, self.W[post_name])
            self.dW[name] += np.dot(delta[:, np.newaxis], self.Z[pre_name][np.newaxis, :])
            self.dB[name] += delta

    def accuracy(self, x, t):
        y = self.forprop(x)
        y = np.argmax(y, axis=1)
        if t.ndim != 1:
            t = np.argmax(t, axis=1)
        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy

    def flush(self):
        # reset the data of dW and dB 
        for l in range(1, self.depth+1):
            name = 'layer{}'.format(l)
            self.dW[name] = np.zeros_like(self.dW[name])
            self.dB[name] = np.zeros_like(self.dB[name])
