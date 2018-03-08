import numpy as np
import time
import pickle
from src.fcnet import FullyConnectedNet
from src.utils.solver import Solver
from src.utils.data_utils import get_FER, pickle_FER

"""
Use a Solver instance to train a TwoLayerNet (FER2013)
"""
def train_FER2013():
  ###########################################################################
  #                           BEGIN OF YOUR CODE                            #
  ###########################################################################
  #pickle_FER() #used to load and store the FER2013 dataset for faster performance
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
                    'learning_rate': 1e-3,#0.0018742807840127864
                    'momentum': 0.5
                  },
                  lr_decay=0.8,
                  num_epochs=100, batch_size=100,
                  print_every=100)
    solver.train()

  #pickle.dump(model, (open("./models/assignment2", "wb"))) #use to save the model after training
##############################################################################
#                             END OF YOUR CODE                               #
##############################################################################