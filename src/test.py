def confusionMatrix(actual, predicted):
  confusion = numpy.zeros(shape=(7,7)).astype(int)
  for i,a in enumerate(actual):
    confusion[actual[i]-1][predicted[i]-1] +=1

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
    classificationrate = 1-(1/numpy.sum(confusion))*(numpy.sum(confusion)-numpy.trace(confusion))


return TP, FP, TN, FN, recallrate, precisionrate, F1,classificationrate


def test_fer_model(img_folder, model="/path/to/model"): #Q5
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