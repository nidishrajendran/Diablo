import os
import pickle
import numpy as np
from scipy import spatial
from dp import doDynamicPooling

def getEmbeddings(line):
	result = []
	for word in line:
		e = word2vecDict[word]
		if len(e)!=100:
			print "\nLocha!"
		result.append(e)
	return result

# Start - Generates pickle file with tuples of Similarity Matrix, label
word2vecDict = pickle.load(open("data/word-embeddings.pickle", "rb" ))
labels = np.loadtxt("data/labels.txt")
tokenized_lines = []

with open('data/sentences.txt', 'r') as corpus:
    for line in corpus:
        tokenized_lines.append(line.strip().split())

# List of similarity matrices for each pair of sentences in the corpus
simMats = []
count = 0
for i in xrange(0, len(tokenized_lines), 2):
	if i%1000==0:
		print "Sentences processed -", i

	# Take every pair of sentences
	s1 = tokenized_lines[i]
	s2 = tokenized_lines[i+1]

	# Get a list of word embeddings for every word in the sentences
	We1 = getEmbeddings(s1)
	We2 = getEmbeddings(s2)

	simMat = np.zeros((len(We1), len(We2)))
	for j in xrange(len(We1)):
		for k in xrange(len(We2)):
			simMat[j,k] = spatial.distance.euclidean( We1[j], We2[k] )
	
	simMat = doDynamicPooling(simMat, 10)
	label = labels[i/2]

	simMats.append((simMat, label))

print len(simMats)
pickle.dump(simMats,open("data/simMats.pickle",'wb'))
