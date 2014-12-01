from __future__ import division
import nltk
import pickle
import os

def loadNormalizationDict(fileName):
	dictionary = {}

	if not os.path.exists('dict_dump.pickle'):
		with open(fileName,'r') as inpFile:
			for line in inpFile:
				oovWord,ivWord = line.split('\t')
				#print oovWord,ivWord
				dictionary[oovWord] = ivWord.rstrip()

		pickle.dump(dictionary,open("dict_dump.pickle",'wb'))
	else:
		dictionary = pickle.load(open( "dict_dump.pickle", "rb" ) )
	#print len(dictionary.keys())
	return dictionary


def normalizeInputs(inputFile,dictionary):
	outFile = 'normalizedInput.txt'
	count = 0
	with open(outFile,'wb') as out:
		with open(inputFile,'r') as inpFile:
			for line in inpFile:
				newLine = line.split()
				for i in xrange(len(newLine)):
					word = newLine[i]
					if word in dictionary:
						count += 1
						newLine[i] = dictionary[word]
						print count,word, dictionary[word]
				newLine = ' '.join(newLine)
				print >> out,newLine



def main():
	dictFile = 'emnlp_dict.txt'
	inputFile = 'input.txt'
	dictionary = loadNormalizationDict(dictFile)
	normalizeInputs(inputFile,dictionary)

if __name__ == "__main__":
	main()