import numpy as np
import time
import pickle
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

out = get_FER(num_training=10000)
data = {
      'X_train': out['X_train'], # training data
      'y_train': out['y_train'], # training labels
      'X_val':  out['X_val'], # validation data
      'y_val': out['y_val'] # validation labels
    }

model = FullyConnectedNet(input_dim=48*48*1, hidden_dims=[40], num_classes=7, dropout=0, reg=0, seed=int(time.time()))
solver = Solver(model, data,
              update_rule='sgd_momentum',
              optim_config={
                'learning_rate': 1e-3,
                'momentum': 0.5
              },
              lr_decay=0.8,
              num_epochs=100, batch_size=100,
              print_every=100)
solver.train()

#pickle.dump(model, (open("./models/assignment2", "wb")))
##############################################################################
#                             END OF YOUR CODE                               #
##############################################################################