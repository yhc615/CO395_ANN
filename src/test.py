import pickle
import os
import matplotlib.image as mpimg
import numpy as np

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

def test_deep_fer_model(img_folder, model="/path/to/model"): #Q6
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

test()