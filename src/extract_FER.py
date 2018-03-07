import matplotlib.image as mpimg
import numpy as np

X_train = []
Y_train = []
X_test = []
Y_test = []

with open("datasets/FER2013/labels_public.txt","r") as labels:
	for line in labels:
		img,emotion = line.split(",")
		if(img.split("/")[0] == "Train"):
			X_train.append(img)
			Y_train.append(int(emotion.split("\n")[0]))
		elif (img.split("/")[0] == "Test"):
			X_test.append(img)
			Y_test.append(int(emotion.split("\n")[0]))
X_train1 = np.asarray(X_train)
X_test1 = np.asarray(X_test)
Y_train1 = np.asarray(Y_train)
Y_test1 = np.asarray(Y_test)
#img = mpimg.imread("datasets/FER2013/"+ XY_train[0][0])

print(X_train1); print(X_test1);print(Y_train1); print(Y_test1)