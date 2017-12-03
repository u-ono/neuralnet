import numpy as np
import pickle
from load_mnist import load_mnist
from PIL import Image
from neuralnet import NeuralNet

def img_show(img):
    _img = Image.fromarray(np.uint8(img))
    _img.show()

# read the dataset
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

# settings of the neuralnet
form = [784, 50, 10]
activ_func = 'relu'
loss_func = 'mean_squared'

# settings of the batch prediction
test_size = x_test.shape[0]

net = NeuralNet(form, activ_func, loss_func)

# settings of the weights parameter
f = open('weights', 'rb')
net.W = pickle.load(f)
f.close()

def check_prediction():
    i = np.random.randint(0, len(x_test))
    img = x_test[i]
    label = np.argmax(t_test[i])
    z = np.argmax(net.forprop(img))
    img_show(img.reshape(28, 28)*255)
    print('label : {}'.format(label))
    print('prediction : {}'.format(z))
    
def prediction_accuracy(batch_size=100):

    # extraction of the batch data
    batch_mask = np.random.choice(test_size, batch_size, replace=False)
    x_batch = x_test[batch_mask]
    t_batch = t_test[batch_mask]

    print('accuracy : {}'.format(net.accuracy(x_batch, t_batch)))