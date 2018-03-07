import numpy as np
import matplotlib.pyplot as plt

from src.fcnet import FullyConnectedNet
from src.utils.solver import Solver
from src.utils.data_utils import get_CIFAR10_data

"""
TODO: Overfit the network with 50 samples of CIFAR-10
"""
###########################################################################
#                           BEGIN OF YOUR CODE                            #
###########################################################################

out = get_CIFAR10_data(num_training=50)
data = {
      'X_train': out['X_train'], # training data
      'y_train': out['y_train'], # training labels
      'X_val':  out['X_val'], # validation data
      'y_val': out['y_val'] # validation labels
    }
model = FullyConnectedNet(hidden_dims=[100], num_classes=10)
solver = Solver(model, data,
                update_rule='sgd',
                optim_config={
                  'learning_rate': 1e-3,
                },
                lr_decay=0.95,
                num_epochs=20, batch_size=25,
                print_every=10)
solver.train()

#-----------------------Plotting--------------------------
plt.subplot(2,1,1)
plt.title("Training loss")
plt.plot(solver.loss_history,"o")
plt.xlabel('Iteration')

plt.subplot(2,1,2)
plt.title('Accuracy')
plt.plot(solver.train_acc_history,'-o', label='train')
plt.plot([0.5]*len(solver.val_acc_history), 'k--')
plt.xlabel('Epoch')
plt.legend(loc='lower right')
plt.gcf().set_size_inches(15,12)
#plt.show()

##############################################################################
#                             END OF YOUR CODE                               #
##############################################################################
