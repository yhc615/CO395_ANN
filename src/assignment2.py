import numpy as np
import time
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

out = get_FER(num_training=1000)
data = {
      'X_train': out['X_train'], # training data
      'y_train': out['y_train'], # training labels
      'X_val':  out['X_val'], # validation data
      'y_val': out['y_val'] # validation labels
    }

reached = False
lr = 1e-2
ws = 1e-2

count = 1
while not reached:
  np.random.seed(int(time.time()))
  print("iteration number{}".format(count))
  ws = 10**(np.random.uniform(-5,-1))
  lr = 10**(np.random.uniform(-3,-2))
  print("testing lr, ws: {},{}".format(lr,ws))
  model = FullyConnectedNet(input_dim=48*48*3, hidden_dims=[100], num_classes=7, dropout=0.5, reg=0.5)#weight_scale=ws)
  solver = Solver(model, data,
                  update_rule='sgd_momentum',
                  optim_config={
                    'learning_rate': lr
                  },
                  lr_decay=0.95,
                  num_epochs=30, batch_size=100,
                  print_every=25)
  solver.train()
  if max(solver.loss_history) == 0.3:
    reached = True
  count += 1

print("Final lr and ws: {},{}".format(lr,ws))
##############################################################################
#                             END OF YOUR CODE                               #
##############################################################################