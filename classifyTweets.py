from __future__ import division
import numpy as np
from sklearn.linear_model import LogisticRegression as LR
from sklearn.svm import LinearSVC as SVM

import pickle
import metrics
from sklearn.metrics import accuracy_score,f1_score
from sklearn.cross_validation import KFold


def loadData(fileName):
	dataDict = pickle.load(open(fileName,'rb'))
	np.random.shuffle(dataDict)
	xData = [a[0].flatten() for a in dataDict]
	yData = [a[1] for a in dataDict]
	return np.array(xData),np.array(yData)


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
	#learner = SVM(penalty='l2',loss='l2')
	learner.fit(xTrain,yTrain)
	return learner

def main():
	for ngms in xrange(1,2):
		fileName = './data/LRsimMats' + str(ngms) +'.pickle'
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

		if F/10 > minF:
			bestClassifier = classifier


	print 'Overall Test Accuracy'
	Y = classifier.predict(xTest)
	print 'Accuracy - ', accuracy_score(yTest,Y)
	print 'P R F  - ', metrics.getPrecisionandRecall(Y,yTest)






if __name__ == '__main__':
	main()