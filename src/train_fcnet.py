import numpy as np

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

out = get_CIFAR10_data(num_training=50,num_validation=20, num_test=0)
data = {
      'X_train': out['X_train'], # training data
      'y_train': out['y_train'], # training labels
      'X_val':  out['X_val'], # validation data
      'y_val': out['y_val'] # validation labels
    }
model = FullyConnectedNet(hidden_dims=[100 for i in range(1)], num_classes=10, dropout=0.5, reg=3.14)
solver = Solver(model, data,
                update_rule='sgd',
                optim_config={
                  'learning_rate': 1.5e-3,
                },
                lr_decay=0.95,
                num_epochs=20, batch_size=2000,
                print_every=5)
solver.train()

##############################################################################
#                             END OF YOUR CODE                               #
##############################################################################
