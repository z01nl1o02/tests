import os,sys,cPickle,pdb
import re

def load_one(infile):
    words = set([])
    with open(infile,'rb') as f:
        for line in f:
            line = line.strip().lower()
            datas = line.split(' ')
            for data in datas:
                if None == re.match(r'[a-z]+$',data):
                    continue
                words.add(data)
    return words

def load(indir,outfile):
    words = set([])
    for root,pdirs,names in os.walk(indir):
        for name in names:
            ext = os.path.splitext(name)[1].lower()
            if ext != '.txt' and ext != '.html':
                print 'skip ',name
                continue
            words = words | load_one(os.path.join(root,name))
    words = sorted( list(words) )
    with open(outfile,'wb') as f:
        f.writelines('\r\n'.join(words))

if __name__=="__main__":
    try:
        os.makedirs('output')
    except Exception,e:
        pass
    load('corpus/train','output/corpus-train.txt')
    load('corpus/test','output/corpus-test.txt')
            

