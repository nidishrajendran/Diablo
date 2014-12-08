python tweetsToInput.py
sed '1~3d' normalizedInput.txt > data/sentences.txt
sed -n 'p;N;N' normalizedInput.txt > data/labels.txt
cd word2vec
python generate_word_vectors.py
