import os,sys,pdb,cPickle
import numpy as np
from sklearn.cluster import KMeans
from toshape import SHAPE_FEAT
import math
import cv2
from collections import defaultdict


def load_one(infile,clf,outfile):
    strokes = []
    results = []
    with open(infile,'rb') as f:
        for line in f:
            label = line.split('|')[0]
            path = line.split('|')[-1]
            datas = line.split('|')[1:-1]
            result = []
            for data in datas:
                cx,cy,ori = [np.float64(x) for x in data.split(',')]             
                result.append( (cx,cy,ori))
            Y = clf.predict( np.asarray(result) )
            result = [label]
            for y in Y:
                result.append( '%d'%(y)  )   
                strokes.append(y)
            result.append(path)
            results.append('|'.join(result))
    with open(outfile,'wb') as f:
        f.writelines('\r\n'.join(results))
    classnum = clf.cluster_centers_.shape[0]
    hist,bins = np.histogram(strokes, normed = True,bins = [x for x in range(classnum)],density=True)
    hist = hist.tolist()
    hist = map(lambda x: x > 0.005, hist)
    proto = []
    for k in range(len(hist)):
        if hist[k] == False:
            continue
        cx = clf.cluster_centers_[k][0]
        cy = clf.cluster_centers_[k][1]
        ori = clf.cluster_centers_[k][2]
        proto.append( (cx,cy,ori))
    outpath = os.path.splitext(outfile)[0] + '.pkl'
    with open(outpath, 'wb') as f:
        cPickle.dump(proto,f)
    return proto
    
def run(indir,outdir):
    with open('strokes.pkl','rb') as f:
        clf = cPickle.load(f)
    try:
        os.makedirs(outdir)
    except Exception,e:
        pass
    for txt in os.listdir(indir):
        load_one( os.path.join(indir,txt) ,clf, os.path.join(outdir,txt)) 
        
        
                
if __name__=="__main__":
    run('feat_norm','proto')
