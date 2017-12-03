import sys, os
sys.path.append(os.pardir)

import numpy as np
from dataset.mnist import load_mnist
from neuralnet import NeuralNet

# read the dataset
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

# settings of the neuralnet
form = [784, 50, 10]
activ_func = 'sigmoid'
loss_func = 'mean_squared'

net = NeuralNet(form, activ_func, loss_func)

# settings of the batch training
it_num = 10000 # number of iterations
train_size = x_train.shape[0]
batch_size = 100
eta = 0.1 # learning rate

# preparation of the outputs
err_list = []
train_accuracy = []
test_accuracy = []

# learning by the neuralnet
it_per_epoch = max(train_size/batch_size, 1)

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
            name = 'layer{}'.format(l)
            net.W[name] -= eta * net.dW[name] / batch_size
            net.B[name] -= eta * net.dB[name] / batch_size
        net.flush()

    # register data per epoch
    if i % it_per_epoch == 0:
        # calculate accuracies
        train_acc = net.accuracy(x_train, t_train)
        test_acc = net.accuracy(x_test, t_test)
        print(train_acc)
        print(test_acc)
        # register data
        err_list.append(delta) 
        train_accuracy.append(train_acc)
        test_accuracy.append(test_acc)


# output weights
weights_file = open('weights', 'w')
weights_file.write(net.W)
weights_file.close()

# output err_list, train_accuracy_list, test_accuracy_list
err_file = open('error', 'w')
err_file.write(err_list)
err_file.close()

train_acc_file = open('train_accuracy', 'w')
train_acc_file.write(train_accuracy)
train_acc_file.close()

test_acc_file = open('test_accuracy', 'w')
test_acc_file.write(test_accuracy)
test_acc_file.close()

# print accuracy in the end
print(test_accuracy.last)
