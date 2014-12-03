from __future__ import division
import numpy as np
from sklearn.linear_model import LogisticRegression as LR
from sklearn.svm import LinearSVC as SVM

import pickle
import metrics
from sklearn.metrics import accuracy_score,f1_score



def loadData():
	dataDict = pickle.load(open('./data/simMats.pickle','rb'))
	np.random.shuffle(dataDict)
	xData = [a[0].flatten() for a in dataDict]
	yData = [a[1] for a in dataDict]
	return xData,yData


def splitData(xData,yData):
	xTrain,xTest = np.split(xData, [int((60/100)*len(xData))])
	yTrain,yTest = np.split(yData, [int((60/100)*len(xData))])

	return xTrain,yTrain,xTest,yTest

def trainClassifier(xTrain,yTrain):
	learner = LR(penalty='l2')
	#learner = SVM(penalty='l2',loss='l2')
	learner.fit(xTrain,yTrain)
	return learner

def main():
	xData,yData = loadData()
	xTrain,yTrain,xTest,yTest = splitData(xData,yData)
	classifier = trainClassifier(xTrain,yTrain)
	Y = classifier.predict(xTest)
	print 'Majority Class Accuracy', accuracy_score(yTest,[0]*len(yTest))
	print 'Accuracy - ', accuracy_score(yTest,Y)
	print 'P R F  - ', metrics.getPrecisionandRecall(Y,yTest)



if __name__ == '__main__':
	main()