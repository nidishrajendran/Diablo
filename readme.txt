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