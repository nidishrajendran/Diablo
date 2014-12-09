Diablo
======
This project is for CS 6601 Artificial Intelligence at Georgia institute of Technology. The project is to Paraphrase tweets, ie find out if two tweets are similar in meaning given the tweets. We implement a sliding window approach, where we learn the word embedding vectors through a neural language model, normalize it, and then run dynamic pooling to get equally sized similarity matrices. We then flatten this and add additional features like Sentence Length, Placeholder word frequency (punctuation, numbers) and Common Named Entity terms to get the final feature vector. We pass this to the Logistic regression classifer and train it to identify similar and non similar sentences from our training set. We achieve a f measure score of 63.8%.

Instructions on how to run: 
To run unnormalized: 

1. Change line 2 and 3 in run.sh to input.txt

sh run.sh
python simMat.py
python classifyTweets.py

To run Normalized:

Check if normalizedInput.txt exists.

If yes:

Change line 2 and 3 in run.sh to normalizedInput.txt instead of input.txt

sh run.sh
python simMat.py
python classifyTweets.py

If no:

python twitterNormalizer.py -- this should generate normalizedInput.txt

Change line 2 and 3 in run.sh to normalizedInput.txt instead of input.txt

sh run.sh
python simMat.py
python classifyTweets.py
