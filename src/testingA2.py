import numpy as np
import time
import pickle
import matplotlib.pyplot as plt
from src.fcnet import FullyConnectedNet
from src.utils.solver import Solver
from src.utils.data_utils import get_FER, pickle_FER

"""
TODO: Use a Solver instance to train a TwoLayerNet that achieves at least 50% 
accuracy on the validation set.
"""
###########################################################################
#                           BEGIN OF YOUR CODE                            #
###########################################################################
#pickle_FER()
optim = False # use for fine tuning learning rate
out = get_FER(num_training=10000)
data = {
      'X_train': out['X_train'], # training data
      'y_train': out['y_train'], # training labels
      'X_val':  out['X_val'], # validation data
      'y_val': out['y_val'] # validation labels
    }
model = FullyConnectedNet(input_dim=48*48*1, hidden_dims=[40], num_classes=7, dropout=0, reg=0, seed=int(time.time()))

if optim:
  count = 1
  reached = False
  threshold = 0.35 #set threshold here
  while not reached:
    np.random.seed(int(time.time()))
    print("iteration number{}".format(count))
    lr = np.random.uniform(0.001,0.003)
    print("testing lr: {}".format(lr))
    solver = Solver(model, data,
                    update_rule='sgd_momentum',
                    optim_config={
                      'learning_rate': lr,
                      'momentum': 0.5
                    },
                    lr_decay=0.8,
                    num_epochs=100, batch_size=100,
                    print_every=100)
    solver.train()
    if max(solver.val_acc_history) >= threshold:
      reached = True
    count += 1

  print("Final lr: {}".format(lr))
else:
  solver = Solver(model, data,
                update_rule='sgd_momentum',
                optim_config={
                  'learning_rate': 0.0018742807840127864,
                  'momentum': 0.5
                },
                lr_decay=0.8,
                num_epochs=100, batch_size=100,
                print_every=100)
  solver.train()

pickle.dump(model, (open("./models/assignment2", "wb")))

plt.subplot(2,1,1)
plt.title("Training loss")
plt.plot(solver.loss_history,"o")
plt.xlabel('Iteration')

plt.subplot(2,1,2)
plt.title('Accuracy')
plt.plot(solver.train_acc_history,'-o', label='train')
plt.plot(solver.val_acc_history,'-o', label='val')
plt.plot([0.5]*len(solver.val_acc_history), 'k--')
plt.xlabel('Epoch')
plt.legend(loc='lower right')
plt.subplots_adjust(.1,.1,.9,.9,.2,.5)
plt.show()
##############################################################################
#                             END OF YOUR CODE                               #
##############################################################################