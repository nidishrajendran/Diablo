import gensim, logging
import scipy.io
import numpy as np
import pickle

filename = "../train.data"
outfile = 'sentences.data'

with open(filename, 'r') as train_file:
    with open(outfile, 'w') as out_file:
        for line in train_file:
            if len(line.strip().split('\t')) == 7:
                (trendid, trendname, origsent, candsent, judge, origsenttag, candsenttag) = line.split('\t')
                out_file.write(origsent + '\n')
                out_file.write(candsent + '\n')
            else:
                continue

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

def read_sentences(filename='sentences.data'):
    sentences = []
    with open(filename, 'r') as sentences_file:
        for line in sentences_file:
            words = line.strip().split()
            sentences.append(words)

    return sentences

sentences = read_sentences()
model = gensim.models.Word2Vec(sentences, min_count=1)

data = {}

word_embeddings = model.syn0.copy()
unknown = np.mean(word_embeddings, axis=0)
word_embeddings = np.vstack((unknown[np.newaxis,:], word_embeddings))
data['We2'] = word_embeddings.T

words = model.index2word[:]
words = ['*UNKNOWN*'] + words
w = np.array(words, dtype='object')
data['words'] = w.reshape((1,w.shape[0]))

data['reIndexMap'] = np.empty((word_embeddings.shape[0],1))

print data['We2'].shape
print data['words'].shape
print data['reIndexMap'].shape

print type(data['We2'])
outDict = {}

for word,val in zip(data['words'][0],data['We2'].T.tolist()):
    outDict[word] = val

pickle.dump(outDict,open('word-embeddings.pickle','wb'))

# temp = pickle.load(open('word-embeddings.pickle','rb'))
# print temp.keys()

scipy.io.savemat('../data/word-embeddings.mat', data)
