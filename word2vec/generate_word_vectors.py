import gensim, logging
import scipy.io
import numpy as np

filename = "train.data"
outfile = './word2vec/sentences.data'

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

def read_sentences(filename='./word2vec/sentences.data'):
    sentences = []
    with open(filename, 'r') as sentences_file:
        for line in sentences_file:
            words = line.strip().split()
            sentences.append(words)

    return sentences

sentences = read_sentences()
model = gensim.models.Word2Vec(sentences, min_count=1)

data = {}
data['We2'] = model.syn0.T

data['words'] = np.array(model.index2word, dtype='object')

data['reIndexMap'] = np.empty((model.syn0.shape[0],1))


print data['We2'].shape
print data['words'].shape
print data['reIndexMap'].shape

print data['words']

scipy.io.savemat('word-embeddings.mat', data)
