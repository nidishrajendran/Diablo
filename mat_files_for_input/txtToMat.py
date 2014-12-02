import numpy as np
import scipy.io

labels = []
sentences = []

for word in open("labels.txt","r"):
	labels.append(int(word))

for line in open("sentences.txt","r"):
	sentences.append(line)

#To get the list of strings as a cell array
my_list = np.zeros((len(sentences),), dtype=np.object)
my_list[:] = sentences

#To get the labels mat
trainSetSize = int(0.7*len(labels))
trainingLabels = np.array(labels[:trainSetSize],dtype=float).T
testingLabels = np.array(labels[trainSetSize:],dtype=float).T

scipy.io.savemat('twitter_all.mat', mdict={'training_labels': trainingLabels,'testing _labels':testingLabels,'sentences':my_list})

