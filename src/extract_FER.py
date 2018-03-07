XY_train, XY_test = [],[]

with open("datasets/FER2013/labels_public.txt","r") as labels:
	for line in labels:
		img,emotion = line.split(",")
		if(img.split("/")[0] == "Train"):
			XY_train.append((img,int(emotion.split("\n")[0])))
		elif (img.split("/")[0] == "Test"):
			XY_test.append((img,int(emotion.split("\n")[0])))
print(XY_train)

