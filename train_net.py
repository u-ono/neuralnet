import pickle
from load_mnist import *
import numpy as np
from neuralnet import NeuralNet

def train():
    for i in range(it_num):
        # extraction of the batch data
        batch_mask = np.random.choice(train_size, batch_size, replace=False)
        x_batch = x_train[batch_mask]
        t_batch = t_train[batch_mask]
    
        for j in range(batch_size):
            z = net.forprop(x_batch[j])
            delta = net.loss(z, t_batch[j])
            net.backprop(delta)
            for l in range(1, net.depth+1):
                net.W[l] -= eta * net.dW[l] / batch_size
                net.B[l] -= eta * net.dB[l] / batch_size
            net.flush()
    
        # register data per epoch
        if i % it_per_epoch == 0:
            # calculate accuracies
            train_acc = net.accuracy(x_train, t_train)
            test_acc = net.accuracy(x_test, t_test)
            print('train_accuracy : {}'.format(train_acc))
            print('test_accuracy : {}'.format(test_acc))
            # register data
            err_list.append(delta) 
            train_accuracy.append(train_acc)
            test_accuracy.append(test_acc)


# read the dataset
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

# settings of the neuralnet
form = [784, 100, 50, 10]
activ_func = ['relu', 'relu', 'softmax']
loss_func = 'cross_entropy'

net = NeuralNet(form, activ_func, loss_func)

# settings of the batch training
it_num = 18000 # number of iterations
train_size = x_train.shape[0]
batch_size = 100
eta = 0.1 # learning rate

# preparation of the outputs
err_list = []
train_accuracy = []
test_accuracy = []

# learning by the neuralnet
it_per_epoch = max(train_size/batch_size, 1)

train()

# output weights
f = open('weights.pkl', 'wb')
pickle.dump(net.W, f)
f.close()

# output err_list, train_accuracy_list, test_accuracy_list
f = open('error.pkl', 'wb')
pickle.dump(err_list, f)
f.close()

f = open('train_accuracy.pkl', 'wb')
pickle.dump(train_accuracy, f)
f.close()

f = open('test_accuracy.pkl', 'wb')
pickle.dump(test_accuracy, f)
f.close()

