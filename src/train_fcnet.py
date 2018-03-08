import numpy as np
import matplotlib.pyplot as plt

from src.fcnet import FullyConnectedNet
from src.utils.solver import Solver
from src.utils.data_utils import get_CIFAR10_data

"""
TODO: Use a Solver instance to train a TwoLayerNet that achieves at least 50% 
accuracy on the validation set.
"""
###########################################################################
#                           BEGIN OF YOUR CODE                            #
###########################################################################

out = get_CIFAR10_data(num_training=25000)
data = {
      'X_train': out['X_train'], # training data
      'y_train': out['y_train'], # training labels
      'X_val':  out['X_val'], # validation data
      'y_val': out['y_val'] # validation labels
    }
model = FullyConnectedNet(hidden_dims=[100], num_classes=10, dropout=0, reg=0.5)
solver = Solver(model, data,
                update_rule='sgd',
                optim_config={
                  'learning_rate': 2e-3,
                },
                lr_decay=0.95,
                num_epochs=25, batch_size=250,
                print_every=100)
solver.train()

#-----------------------Plotting--------------------------
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
