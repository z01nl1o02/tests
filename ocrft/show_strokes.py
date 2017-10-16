import os,sys,pdb,cPickle
import numpy as np
from sklearn.cluster import KMeans
from toshape import SHAPE_FEAT
import math
import cv2




def load_one(infile,clf):
    results = []
    with open(infile,'rb') as f:
        for line in f:
            datas = line.split('|')[1:-1]
            result = []
            for data in datas:
                cx,cy,ori = [np.float64(x) for x in data.split(',')]             
                result.append( (cx,cy,ori))
            Y = clf.predict( np.asarray(result) )
            result = []
            for y in Y:
                result.append( clf.cluster_centers_[y] )
            results.append(result)
    return results
    
def run(indir,outdir,w = 32,h = 64):
    with open('strokes.pkl','rb') as f:
        clf = cPickle.load(f)
    folderset = set([])
    sft = SHAPE_FEAT(False,8,8)
    idx = 0
    for txt in os.listdir(indir):
        shapes =  load_one( os.path.join(indir,txt) ,clf) 
        cid = txt.split('.')[0]
        odir = os.path.join(outdir,cid)
        if odir not in folderset:
            os.makedirs(odir)
            folderset.add(odir)
        num = 0
        for shape in shapes:
            img = np.zeros((h * 2,w * 2),np.uint8)
            shape_resized = []
            for stroke in shape:
                cx = np.int64(stroke[0] * w + w/2)
                cy = np.int64(stroke[1] * h + h/2)
                ori = stroke[2] * math.pi
                shape_resized.append( (cx,cy,ori) )
            img = sft.draw_shape(shape_resized,img)
            idx += 1
            outpath = os.path.join(odir,'%s_%d.jpg'%(cid,idx))
            cv2.imwrite(outpath,img)
            num += 1
            if num > 10:
                break
                
if __name__=="__main__":
    run('feat_norm','show')
