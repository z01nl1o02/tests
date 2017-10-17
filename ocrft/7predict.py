import os,sys,pdb,cPickle
import numpy as np
from sklearn.cluster import KMeans
from toshape import SHAPE_FEAT
import math
import cv2
from collections import defaultdict
import math
def load_proto(indir):
    shapes = []
    labels = []
    for pkl in os.listdir(indir):
        if os.path.splitext(pkl)[1] != '.pkl':
            continue
        with open( os.path.join(indir,pkl), 'rb') as f:
            strokes = cPickle.load(f)
        cid = os.path.splitext(pkl)[0]
        labels.append( cid )
        shapes.append( strokes )
    return (labels,shapes)

def load_one_test(infile):
    labels = []
    paths = []
    shapes = []
    with open(infile,'rb') as f:
        for line in f:
            data = line.strip().split('|')
            label = data[0]
            path = data[-1]
            shape = []
            for stroke in data[1:-1]:
                cx,cy,ori = [np.float64(x) for x in stroke.split(',')]
                shape.append( (cx,cy,ori) )
            labels.append(label)
            paths.append(path)
            shapes.append(shape)
    return (labels,shapes,paths)

def calc_stroke_distance_ovo(a,b):
    d0 = a[0] - b[0]
    d1 = a[1] - b[1]
    d2 = a[2] - b[2]
    return math.sqrt( d0 ** 2 + d1 ** 2 + d2 ** 2)
    
    
def calc_stroke_distance_ova(t,shape1):
    dist = map( lambda x: calc_stroke_distance_ovo(t,x), shape1)
    return reduce(lambda a,b: np.minimum(a,b),dist)
    
def calc_shape_distance(shape0,shape1,K=1):
    Ef = []
    for x in shape0:
        Ef.append( calc_stroke_distance_ova(x,shape1) )
    Ep = []
    for x in shape1:
        Ep.append( calc_stroke_distance_ova(x,shape0) )

    Ef = map(lambda d: 1.0 / (1 + K*(d**2)), Ef)
    Ep = map(lambda d: 1.0 / (1 + K*(d**2)), Ep)
    
    EpSum = np.asarray(Ep).sum()
    EfSum = np.asarray(Ef).sum()
    A = EpSum + EfSum
    B = len(Ep) + len(Ef)
    return 1 - (A * 1.0 / B)
    
def predict(protolabels,protoshapes,labels,shapes):
    hit = 0
    for label, shape in zip(labels,shapes):
        dist = []
        for pla,psh in zip(protolabels,protoshapes): 
            dist.append( calc_shape_distance(shape,psh) )
        k = np.argmin(dist)
        if label == protolabels[k]:
            hit += 1
    return hit * 1.0 / len(labels)


    
    
def run(indir,protodir):
    try:
        os.makedirs(outdir)
    except Exception,e:
        pass
    with open('strokes.pkl','rb') as f:
        clf = cPickle.load(f)
    protolabels, protoshapes = load_proto(protodir)

    for txt in os.listdir(indir):
        labels,shapes,paths = load_one_test( os.path.join(indir,txt) ) 
        recalling = predict(protolabels,protoshapes,labels,shapes)
        print txt,',',recalling 
        
                
if __name__=="__main__":
    run('feat_norm','proto')
