X = []
Y=[]

with open("../datasets/FER2013/labels_public.txt","r") as labels:
	i = 0
	for line in labels:
		img,emotion = line.split(",")
		if i>0:
			X.append(img)
			Y.append(emotion)
			print(Y)
		i+=1

print(train)
