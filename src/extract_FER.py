X_train, X_test = [],[]
Y_train, Y_test = [],[]

with open("datasets/FER2013/labels_public.txt","r") as labels:
	for line in labels:
		img,emotion = line.split(",")
		if(img.split("/")[0] == "Train"):
			X_train.append(img)
			Y_train.append(int(emotion.split("\n")[0]))
		elif (img.split("/")[0] == "Test"):
			X_test.append(img)
			Y_test.append(int(emotion.split("\n")[0]))
print(Y_train)

