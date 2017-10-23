import os,sys,pdb,cPickle
from collections import defaultdict
import numpy as np
def load_corpus(infile):
    words = []
    with open(infile,'rb') as f:
        for line in f:
            word = '-' + line.strip() + '-' #check heading/tailing letter distribution
            words.append(word)
    return words

def build_n_gram(words,n):
    model = defaultdict(np.float64)
    for word in words:
        size = len(word)
        if size <  n - 1:
            continue
        for k in range(0,size-n,1):
            model[word[k:k+n]] += 1.0
    total = 0
    for key in model.keys():
        total += model[key]
    for key in model.keys():
        model[key] = model[key]/total
    return model

def run(corpus_file, model_file):
    words = load_corpus(corpus_file)
    n1gram = build_n_gram(words,1)
    n2gram = build_n_gram(words,2)
    n3gram = build_n_gram(words,3)
    with open(model_file,'wb') as f:
        cPickle.dump((n1gram,n2gram,n3gram),f)
    return

if __name__=="__main__":
    run('output/corpus-train.txt','output/model.pkl')


