filename = "train.data"
out = open('input.txt', 'w')

for line in open(filename):
    if len(line.strip().split('\t')) == 7:
        (trendid, trendname, origsent, candsent, judge, origsenttag, candsenttag) = line.split('\t')
    else:
        continue
    
    if origsent == candsent:
        continue

    # ignoring the training/test data that has middle label 
    if judge[0] == '(':
        nYes = eval(judge)[0]            
        if nYes >= 3:
            out.write("1\n")
        elif nYes <= 1:
            out.write("0\n")
        else:
            continue
    out.write(origsent + "\n")
    out.write(candsent + "\n")
out.close()