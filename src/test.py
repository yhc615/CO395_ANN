import os
import time
import numpy as np
import matplotlib.image as mpimg
import pickle

from src.fcnet import FullyConnectedNet
from src.utils.solver import Solver
from src.utils.data_utils import get_FER, pickle_FER

def confusionMatrix(actual, predicted):
    confusion = np.zeros(shape=(7,7)).astype(int)
    for i,a in enumerate(actual):
        confusion[actual[i]][predicted[i]] +=1
    print(confusion)
    return confusion

def conMatStats(confusion):
    recallrate, precisionrate,F1 = [0]*7,[0]*7,[0]*7
  
    TP,FP,TN,FN =[0]*7,[0]*7,[0]*7,[0]*7
  
    for i in range(7):
        for j in range(7):
            TP[i] = confusion[i][i]
            if j != i:
                FN[i] += confusion[i][j]
        
                FP[i] += confusion[j][i]
                for k in range(7):
                    if k!=i:
                        TN[i] += confusion[j][k]

    for i in range(7):
        if(TP[i] != 0 or FN[i] !=0):
            recallrate[i] = float(TP[i])/(TP[i]+FN[i])
        if(TP[i] != 0 or FP[i] !=0):
            precisionrate[i] = float(TP[i])/(TP[i]+FP[i])
        if(precisionrate[i] !=0 or recallrate[i] != 0):
            F1[i] = 2*(precisionrate[i]*recallrate[i])/(precisionrate[i]+recallrate[i])
    classificationrate = 1-(1/np.sum(confusion))*(np.sum(confusion)-np.trace(confusion))

    return TP, FP, TN, FN, recallrate, precisionrate, F1, classificationrate

def test_fer_model(img_folder, model="./models/assignment2"): #Q5
    """
    Given a folder with images, load the images (in lexico-graphical ordering
    according to the file name of the images) and your best model to predict
    the facial expression of each image.
    Args:
    - img_folder: Path to the images to be tested
    Returns:
    - preds: A numpy vector of size N with N being the number of images in
    img_folder.
    """
    preds = None
    ### Start your code here
    modelDat = pickle.load(open(model, "rb"))
    X = []
    for filename in os.listdir(img_folder):
        if filename.endswith(".jpg"): 
            img = mpimg.imread(img_folder + "/" + filename)
            X.append(img)

    X = np.asarray(X).astype("float")
    subtract_mean = True
    if subtract_mean:
        mean_image = np.mean(X, axis=0)
        X -= mean_image
    X = X[:,:,:,0]

    scores = modelDat.loss(X)
    preds = np.argmax(scores, axis=1)
    
    ### End of code
    return preds

def test():
    Y=[]
    i = 0
    preds = test_fer_model("./datasets/FER2013/Test/", "./models/assignment2")
    with open("datasets/FER2013/labels_public.txt","r") as labels:
        Y = []
        for line in labels:
            img,emotion = line.split(",")
            if (img.split("/")[0] == "Test"):
                Y.append(int(emotion.split("\n")[0]))
    print(i)
    conMat = confusionMatrix(Y,preds)
    for row in conMatStats(conMat):
      print(row)

#test()

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
#train_FER2013() use to train the FER2013 model
def test_deep_fer_model(img_folder, model="/path/to/model"): #Q6 Not Implemented
    """
    Given a folder with images, load the images (in lexico-graphical ordering
    according to the file name of the images) and your best model to predict
    the facial expression of each image.
    Args:
    - img_folder: Path to the images to be tested
    Returns:
    - preds: A numpy vector of size N with N being the number of images in
    img_folder.
    """
    preds = None
    ### Start your code here
    pass
    ### End of code
    return preds
