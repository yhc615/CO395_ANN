import matplotlib.image as mpimg
import numpy as np

def load_FER():#loads whole FER dataset
	X_train = []
	Y_train = []
	X_test = []
	Y_test = []

	with open("datasets/FER2013/labels_public.txt","r") as labels:
		for line in labels:
			img,emotion = line.split(",")
			if(img.split("/")[0] == "Train"):
				img_data = mpimg.imread("datasets/FER2013/"+ img)
				print(img_data)
				X_train.append(img_data)
				Y_train.append(int(emotion.split("\n")[0]))
			elif (img.split("/")[0] == "Test"):
				img_data = mpimg.imread("datasets/FER2013/"+ img)
				X_test.append(img_data)
				Y_test.append(int(emotion.split("\n")[0]))
	return {
	'X_train' : np.asarray(X_train),
	'X_test' : np.asarray(X_test),
	'Y_train' : np.asarray(Y_train),
	'Y_test' : np.asarray(Y_test)
	}

if __name__ == '__main__':
	print (load_FER())


#img = mpimg.imread("datasets/FER2013/"+ XY_train[0][0])
