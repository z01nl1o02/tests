import os,sys,pdb,cPickle
import numpy as np
from sklearn.cluster import KMeans
from toshape import SHAPE_FEAT
import math
import cv2




def load_one(infile,clf,outfile):
    results = []
    with open(infile,'rb') as f:
        for line in f:
            datas = line.split('|')[1:-1]
            label = datas[0]
            path = datas[-1]
            result = []
            for data in datas:
                cx,cy,ori = [np.float64(x) for x in data.split(',')]             
                result.append( (cx,cy,ori))
            Y = clf.predict( np.asarray(result) )
            result = [label]
            for y in Y:
                cx,cy,ori = clf.cluster_centers_[y]
                result.append( '%f,%f,%f'%(cx,cy,ori)  )           
            result.append(path)
            results.append('|'.join(result))
    with open(outfile,'wb') as f:
        f.writelines('\r\n'.join(results))
    return results
    
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
