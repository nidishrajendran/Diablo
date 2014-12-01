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

        
