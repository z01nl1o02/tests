import os,sys,pdb,cPickle
from sklearn.cluster import KMeans
import numpy as np


def load_one(infile):
    results = []
    with open(infile,'rb') as f:
        for line in f:
            datas = line.split('|')[1:-1]
            for data in datas:
                cx,cy,ori = [np.float64(x) for x in data.split(',')]
                results.append((cx,cy,ori))
    return results

def run(indir):
    X = []
    for txt in os.listdir(indir):
        X.extend( load_one( os.path.join(indir,txt) ) )
    print len(X)
    clf = KMeans(100).fit( np.asarray(X) )
    with open('strokes.pkl','wb') as f:
        cPickle.dump(clf, f)

if __name__=="__main__":
    run('feat_norm')

