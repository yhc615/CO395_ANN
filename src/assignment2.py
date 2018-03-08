import numpy as np

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
pickle_FER()

out = get_FER(num_training=25000)
data = {
      'X_train': out['X_train'], # training data
      'y_train': out['y_train'], # training labels
      'X_val':  out['X_val'], # validation data
      'y_val': out['y_val'] # validation labels
    }

reached = False
lr = 1e-2
ws = 1e-2

while not reached:
  ws = 1**(np.random.uniform(-5,-1))
  lr = 1**(np.random.uniform(-4,-1))
  model = FullyConnectedNet(input_dim=48*48*3, hidden_dims=[100], num_classes=7, dropout=0, reg=0.5, weight_scale=ws)
  solver = Solver(model, data,
                  update_rule='sgd_momentum',
                  optim_config={
                    'learning_rate': lr
                  },
                  lr_decay=0.95,
                  num_epochs=20, batch_size=250,
                  print_every=100)
  solver.train()
  if max(solver.loss_history) == 1.0:
    reached = True

print("Final lr and ws: {},{}".format(lr,ws))
##############################################################################
#                             END OF YOUR CODE                               #
##############################################################################