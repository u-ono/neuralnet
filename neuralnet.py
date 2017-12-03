import numpy as np
from layers import *

class NeuralNet:

    def __init__(self, form, activ_func, loss_func, std = 0.01):
        # depth of the neuralnet
        self.depth = len(form)-1

        # settings of W, B, dW, dB
        self.W = [None]*(self.depth+1)
        self.B = [None]*(self.depth+1)
        self.dW = [None]*(self.depth+1)
        self.dB = [None]*(self.depth+1)
        for l in range(1, self.depth+1):
            self.W[l] = std * np.random.randn(form[l], form[l-1])
            self.dW[l] = np.zeros((form[l], form[l-1]))
            self.B[l] = std * np.random.randn(form[l])
            self.dB[l] = np.zeros(form[l])

        # initialization of Z(outputs of each activation layer)
        self.Z = [None]*(self.depth+1)
        # initialization of U(outputs of each affine layer)
        self.U = [None]*(self.depth+1)

        # settings of the activation function
        if activ_func == 'sigmoid':
            self.h = Sigmoid(1.0)
        elif activ_func == 'relu':
            self.h = ReLU()
        else:
            print("activate fanction must be 'sigmoid' or 'relu'")

        # settings of the loss function

    def forprop(self, x):

        z = x
        self.Z[0] = z
        for l in range(1, self.depth+1):
            u = np.dot(z, self.W[l].T)
            u += self.B[l]
            self.U[l] = u
            z = self.h.forward(u)
            self.Z[l] = z

        return z
        
    def loss(self, z, t):

        delta = (z - t) * 1
        dW = np.dot(delta[:, np.newaxis], self.Z[self.depth-1][np.newaxis, :])
        self.dW[self.depth] += dW
        self.dB[self.depth] += delta

        return delta

    def backprop(self, delta):

        for l in range(self.depth-1, 0, -1):
            delta = self.h.backward(self.U[l]) * np.dot(delta, self.W[l+1])
            dW = np.dot(delta[:, np.newaxis], self.Z[l-1][np.newaxis, :])
            self.dW[l] += dW
            self.dB[l] += delta

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
            self.dW[l] = np.zeros_like(self.dW[l])
            self.dB[l] = np.zeros_like(self.dB[l])
