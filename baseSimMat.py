import os
import pickle
import cPickle
import numpy as np
from scipy import spatial
from dp import doDynamicPooling
from nltk.util import ngrams

labels = np.loadtxt("data/labels.txt")
tokenized_lines = []

with open("data/sentences.txt", "r") as corpus:
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

	simMat = np.zeros((len(s1), len(s2)))
	for j in xrange(len(s1)):
		for k in xrange(len(s2)):
			if s1[j] == s2[k]:
				simMat[j,k] = 1

	
	simMat = doDynamicPooling(simMat, 11)
	label = labels[i/2]

	simMats.append((simMat, label))

print len(simMats)
pickle.dump(simMats,open("data/simMats15.pickle",'wb'))
pickle.dump(tokenized_lines,open("data/tokenized_lines"+str(15)+".pickle",'wb'))