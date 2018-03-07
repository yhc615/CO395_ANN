import matplotlib.image as mpimg
import numpy as np

X_train = np.array([])
Y_train = np.array([]) 
X_test = np.array([])
Y_test = np.array([])

with open("datasets/FER2013/labels_public.txt","r") as labels:
	for line in labels:
		img,emotion = line.split(",")
		if(img.split("/")[0] == "Train"):
			X_train = np.append(X_train,img)
			Y_train = np.append(Y_train, int(emotion.split("\n")[0]))
		elif (img.split("/")[0] == "Test"):
			X_test = np.append(X_test,img)
			Y_test = np.append(Y_test, int(emotion.split("\n")[0]))
print(X_test)
#img = mpimg.imread("datasets/FER2013/"+ XY_train[0][0])

#print(img)