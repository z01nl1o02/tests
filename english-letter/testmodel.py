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

def load_model(model_file):
    with open(model_file,'rb') as f:
        ngrams = cPickle.load(f)
    return ngrams

def test_one_word(word,model,n):
    size = len(word)
    if size + 1 < n:
        return False
    logP = 0
    for k in range(0,size - n,1):
        p0 = word[k:k+n-1]
        p1 = word[k:k+n]
        p0 = model[n-2][p0]
        p1 = model[n-1][p1]
        if p0 < 1e-10:
            p0 = 1e-10
        if p1 < 1e-10:
            p1 = 1e-10
        logP = logP + ( np.log10(p1) - np.log10(p0) )
    if logP < np.log10( 10e-8 ):
        return False
    return True
        
        

def run(corpus_file, model_file, result_file):
    words = load_corpus(corpus_file)
    ngrams = load_model(model_file)
    results = []
    for word in words:
        if True == test_one_word(word,ngrams,3):
            continue
        results.append(word.strip('-'))
    with open(result_file,'wb') as f:
        f.writelines('\r\n'.join(results))
    return

if __name__=="__main__":
    run('output/corpus-test.txt','output/model.pkl', 'output/result.txt')




