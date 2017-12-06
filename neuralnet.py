import numpy as np
from layers import *

class NeuralNet:

    def __init__(self, form, activ_func, loss_func, std = 0.01):
        # depth of the neuralnet
        self.depth = len(form)-1
        # last layer's name
        self.last_layer = activ_func[-1]
        # settings of W, B, dW, dB, h
        self.W = [None]*(self.depth+1)
        self.B = [None]*(self.depth+1)
        self.dW = [None]*(self.depth+1)
        self.dB = [None]*(self.depth+1)
        self.h = [None]*(self.depth+1)
        self.h[0] = Identity(form[0])
        for l in range(1, self.depth+1):
            # settings of each layer's W and B
            self.W[l] = std * np.random.randn(form[l], form[l-1])
            self.dW[l] = np.zeros((form[l], form[l-1]))
            self.B[l] = std * np.random.randn(form[l])
            self.dB[l] = np.zeros(form[l])
            # settings of the activation function
            if activ_func[l-1] == 'sigmoid':
                self.h[l] = Sigmoid(form[l], 1.0)
            elif activ_func[l-1] == 'relu':
                self.h[l] = ReLU(form[l])
            elif activ_func[l-1] == 'identity':
                self.h[l] = Identity(form[l])
            elif activ_func[l-1] == 'softmax':
                self.h[l] = Softmax(form[l])
            else:
                print("activate function error!")
                return 
        # settings of the loss function
        if loss_func == 'mean_squared':
            self.loss_func = MeanSquaredError()
        elif loss_func == 'cross_entropy':
            self.loss_func = CrossEntropyError()
        else:
            print("loss function error!")
            return

    def forprop(self, x):
        z = x
        self.h[0].z = z
        for l in range(1, self.depth+1):
            u = np.dot(z, self.W[l].T)
            u += self.B[l]
            z = self.h[l].forward(u)
        return z
        
    def loss(self, z, t):
        return self.loss_func.forward(z, t)

    def backprop(self, z, t):
        if (self.last_layer == 'softmax' and self.loss_func.name == 'cross_entropy') or (self.last_layer == 'identity' and self.loss_func.name == 'mean_squared'):
            # calculate delta simply
            delta = z - t 
        else:
            # delta_i = SUM{k}(dE/dz_k * dz_k/du_i)
            delta = np.dot(self.loss_func.dE, self.h[self.depth].dZ())
        dW = np.dot(delta[:, np.newaxis], self.h[self.depth-1].z[np.newaxis, :])
        self.dW[self.depth] += dW
        self.dB[self.depth] += delta
        for l in range(self.depth-1, 0, -1):
            delta = self.h[l].diff() * np.dot(delta, self.W[l+1])
            dW = np.dot(delta[:, np.newaxis], self.h[l-1].z[l-1])
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
