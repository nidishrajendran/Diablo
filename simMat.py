import os
import pickle
import cPickle
import numpy as np
from scipy import spatial
from dp import doDynamicPooling
from nltk.util import ngrams

wikiVectors = 1
ngms = 2


def getKEmbeddings(line,ngms):
	global averageWe
	while len(line) < ngms:
		line += line[-1]
	grams = ngrams(line,ngms)
	result = []
	for gram in grams:
		e  = []
		for word in gram:
			if word in word2vecDict:
				e = np.hstack((e,word2vecDict[word]))
			else:
				e = np.hstack((e,averageWe))
				
		result.append(e)
	return result


def computeDict():
	with open("data/wikiVectors.pkl", "r") as f:
		raw = cPickle.load(f)
	words, vecs = raw

	dic = {}
	for i in xrange(len(words)):
		dic[words[i]] = vecs[i,:]

	return dic, np.sum(vecs, axis=0)/float(vecs.shape[0])


# Start - Generates pickle file with tuples of Similarity Matrix, label
if wikiVectors==1:
	word2vecDict, averageWe = computeDict()
else:
	word2vecDict, averageWe = pickle.load(open("data/word-embeddings.pickle", "rb" )), None

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

	# Get a list of word embeddings for every word in the sentences
	We1 = getKEmbeddings(s1,ngms)
	We2 = getKEmbeddings(s2,ngms)

	simMat = np.zeros((len(We1), len(We2)))
	for j in xrange(len(We1)):
		for k in xrange(len(We2)):
			simMat[j,k] = spatial.distance.euclidean( We1[j], We2[k] )
	
	simMat = doDynamicPooling(simMat, 5)
	label = labels[i/2]

	simMats.append((simMat, label))

print len(simMats)
print len(labels[labels == 1])
pickle.dump(simMats,open("data/vPoolData/NsimMats" + str(15) + ".pickle",'wb'))
