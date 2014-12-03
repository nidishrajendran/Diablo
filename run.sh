python tweetsToInput.py
sed '1~3d' input.txt > data/sentences.txt
sed -n 'p;N;N' input.txt > data/labels.txt
cd word2vec
python generate_word_vectors.py
cd ..
python simMat.py
python classifyTweets.py