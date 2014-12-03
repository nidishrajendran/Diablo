import os
import pickle
import numpy as np
from scipy import spatial

def getEmbeddings(line):
	result = []
	for word in line:
		e = word2vecDict[word]
		if len(e)!=100:
			print "\nLocha!"
		result.append(e)
	return result

# Start
word2vecDict = pickle.load(open("data/word-embeddings.pickle", "rb" ))
tokenized_lines = []

with open('data/sentences.data', 'r') as corpus:
    for line in corpus:
        tokenized_lines.append(line.strip().split())

# List of similarity matrices for each pair of sentences in the corpus
simMats = []
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
	for i in xrange(len(We1)):
		for j in xrange(len(We2)):
			simMat[i,j] = spatial.distance.euclidean( We1[i], We2[j] )
	simMats.append(simMat)

pickle.dump(simMats,open("data/simMats.pickle",'wb'))
