import numpy as np
import scipy.io

labels = []
sentences = []
for word in open("labels.txt","r"):
	labels.append(int(word))

for line in open("sentences.txt","r"):
	sentences.append(line)


trainSetSize = int(0.7*len(labels))
trainingLabels = np.array(labels[:trainSetSize])
testingLabels = np.array(labels[trainSetSize:])

scipy.io.savemat('twitter_all.mat', mdict={'trainingLabels': trainingLabels,'testingLabels':testingLabels,'sentences':sentences})

