import numpy as np
import pickle
import matplotlib.pyplot as plt

f = open('train_accuracy.pkl', 'rb')
train_acc = pickle.load(f)
f.close()

f = open('test_accuracy.pkl', 'rb')
test_acc = pickle.load(f)
f.close()

fig = plt.figure()

ax1 = fig.add_subplot(121)
ax1.plot(np.arange(1, len(train_acc)+1), train_acc)
ax1.grid(True)
ax1.set_xlabel('epoch')
ax1.set_ylabel('accuracy')
ax1.set_ylabel('train data')

ax2 = fig.add_subplot(122)
ax2.plot(np.arange(1, len(test_acc)+1), test_acc)
ax2.grid(True)
ax2.set_xlabel('epoch')
ax2.set_ylabel('accuracy')
ax2.set_ylabel('test data')

fig.tight_layout()
plt.show()
fig.savefig('softmax.png')
