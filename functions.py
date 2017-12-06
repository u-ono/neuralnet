import numpy as np

def sigmoid(a, x):
    return 1.0 / (1 + np.exp(-a*x))

def sigmoid_diff(a, x):
    return a * (1.0 -sigmoid(a, x)) * sigmoid(a, x)

def relu(x):
    return np.maximum(0, x)

def relu_diff(x):
    diff = np.zeros_like(x)
    diff[x>=0] = 1
    return diff

def softmax(x):
    if x.ndim == 2:
        x = x.T
        x = x - np.max(x, axis=0)
        y = np.exp(x)/np.sum(np.exp(x), axis=0)
        y = y.T
    else:
        x = x - np.max(x)
        y = np.exp(x)/np.sum(np.exp(x))
    return y

def mean_squared_error(y, t):
    return 0.5 * np.sum((y-t)**2)

'''
def cross_entropy_error(y, t):

    if t.shape != y.shape:
        t_ = np.zeros_like(y)
        for i in range(len(t)):
            t_[i, t[i]] = 1

    if y.ndim == 1:
        y = y.reshape(1, y.size)
        t_ = t_.reshape(1, t.size)

    batch_size = y.shape[0]

    return -np.sum(np.log(y)*t) / batch_size
'''

def cross_entropy_error(y, t):
    return - np.sum(t * np.log(y))
