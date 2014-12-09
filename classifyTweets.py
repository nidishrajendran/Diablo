from __future__ import division
import numpy as np
from sklearn.linear_model import LogisticRegression as LR
from sklearn.svm import LinearSVC as SVM

import pickle
import metrics
from sklearn.metrics import accuracy_score,f1_score
from sklearn.cross_validation import KFold
from random import shuffle

from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import re
from collections import defaultdict

stop = stopwords.words('english')
lmtzr = PorterStemmer()
numberPattern = re.compile("[0-9]+")
punctpattern = re.compile("[?!.-;,:]+")

def loadData(fileName):
	dataDict = pickle.load(open(fileName,'rb'))
	tokenized_lines = pickle.load(open("data/tokenized_lines15.pickle","rb"))
#	np.random.shuffle(dataDict)
	xData = [a[0].flatten() for a in dataDict]
	yData = [a[1] for a in dataDict]
	tempData = []
	titleFeature = []
	lengthFeature = []
	cwFeature = []
	punctFeature = []
	numbFeature = []
	for i in xrange(0, len(tokenized_lines), 2):
		# Take every pair of sentences
		s1 = tokenized_lines[i]
		s2 = tokenized_lines[i+1]
		features = defaultdict()
		features[1] = defaultdict(int)
		features[2] = defaultdict(int)
		st1 = set()
		cw = 0
		for word in s1:
			if word.istitle() or word.isupper():
				features[1]['title']+=1
		#	if word.lower() not in stop:
			st1.add(word)
			if punctpattern.match(word)!=None:
				features[1]['punct']+=1
			if numberPattern.match(word)!=None:
				features[1]['numb']+=1
		for word in s2:
			if word.istitle() or word.isupper():
				features[2]['title']+=1
			if punctpattern.match(word)!=None:
				features[2]['punct']+=1
			if numberPattern.match(word)!=None:
				features[2]['numb']+=1
		#	if lmtzr.stem(word.lower()) in st1:
			if word in st1:
				cw+=1
			
	#	print t1,t2
	#	raw_input()
		titleFeature.append((features[1]['title'],features[2]['title']))
		lengthFeature.append((len(s1),len(s2)))
		punctFeature.append((features[1]['punct'],features[2]['punct']))
		numbFeature.append((features[1]['numb'],features[2]['numb']))
		cwFeature.append(cw)
	#	atFeature.append((ind1,ind2))
	print len(titleFeature)
	for i in xrange(len(xData)):
		x = list(xData[i])
		x.append(abs(titleFeature[i][0]-titleFeature[i][1]))
		x.append(abs(lengthFeature[i][0] - lengthFeature[i][1]))
		x.append(cwFeature[i])
		x.append(abs(numbFeature[i][0]-numbFeature[i][1]))
		x.append(abs(punctFeature[i][0] - punctFeature[i][1]))

		n = np.array(x)
		tempData.append((n,yData[i]))
	
	np.random.shuffle(tempData)
	x1Data = []
	y1Data = []
	for i in xrange(len(tempData)):
		x1Data.append(tempData[i][0])
		y1Data.append(tempData[i][1])
#	raw_input()
	return np.array(x1Data),np.array(y1Data)


def splitData(xData,yData):
	kf = KFold(len(xData), n_folds=10)
	for train,test in kf:
		xTrain,yTrain,xTest,yTest = xData[train],yData[train],xData[test],yData[test]
		yield xTrain,yTrain,xTest,yTest

def testTrainSplit(xData,yData):

	xTrain,xTest = np.split(xData, [int((60/100)*len(xData))])
	yTrain,yTest = np.split(yData, [int((60/100)*len(xData))])

	return xTrain,yTrain,xTest,yTest	

def trainClassifier(xTrain,yTrain):
	learner = LR(penalty='l2')
#	learner = SVM(penalty='l2',loss='l2')
	learner.fit(xTrain,yTrain)
	return learner

def main():
	fm = []
	fileName = './data/simMats' + str(15) +'.pickle'
	for j in xrange(10):
		xData,yData = loadData(fileName)
		xTrain,yTrain,xTest,yTest = testTrainSplit(xData,yData)
		print fileName
		bestClassifier = None
		minF = 0
		P = R = F = 0
		for xCVTrain,yCVTrain,xCVTest,yCVTest in splitData(xTrain,yTrain):
			#xTrain,yTrain,xTest,yTest = splitData(xData,yData)
			classifier = trainClassifier(xCVTrain,yCVTrain)
			Y_1 = classifier.predict(xCVTest)
			ret = metrics.getPrecisionandRecall(Y_1,yCVTest)
			P += ret[0]
			R += ret[1]
			F += ret[2]

		print 'Accuracy - ', accuracy_score(yCVTest,Y_1)
		print 'P R F  - ', P/10,R/10,F/10

	#	if F/10 > minF:
	#		bestClassifier = classifier


		print 'Overall Test Accuracy'
		Y = classifier.predict(xTest)
		print 'Accuracy - ', accuracy_score(yTest,Y)
		print 'P R F  - ', metrics.getPrecisionandRecall(Y,yTest)
		fm.append(metrics.getPrecisionandRecall(Y,yTest)[2])
	print 'Average fmeasure',sum(fm)/len(fm)




if __name__ == '__main__':
	main()