import numpy as np
import scipy.io

labels = []
sentences = []

for word in open("labels.txt","r"):
	labels.append(int(word))

for line in open("sentences.txt","r"):
	sentences.append(line)

my_list = np.zeros((len(sentences),), dtype=np.object)
my_list[:] = sentences
trainSetSize = int(0.7*len(labels))
trainingLabels = np.array(labels[:trainSetSize],dtype=float).T
testingLabels = np.array(labels[trainSetSize:],dtype=float).T

scipy.io.savemat('twitter_all.mat', mdict={'trainingLabels': trainingLabels,'testingLabels':testingLabels,'sentences':my_list})

