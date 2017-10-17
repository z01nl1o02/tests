import os,sys,pdb,cPickle
import numpy as np
from sklearn.cluster import KMeans
from toshape import SHAPE_FEAT
import math
import cv2




def load_one(infile):
    with open(infile, 'rb') as f:
        strokes = cPickle.load(f)
    return strokes
    
def run(indir,outdir,w = 32*3,h = 64*3):
    try:
        os.makedirs(outdir)
    except Exception,e:
        print e
        pass
    folderset = set([])
    sft = SHAPE_FEAT(False,8*3,8*3)
    for pkl in os.listdir(indir):
        if os.path.splitext(pkl)[1] != '.pkl':
            continue
        
        strokes =  load_one( os.path.join(indir,pkl)) 
        cid = os.path.splitext(pkl)[0]

        img = np.zeros((h * 2,w * 2),np.uint8)
        shape_resized = []
        for stroke in strokes:
            cx = np.int64(stroke[0] * w + w/2)
            cy = np.int64(stroke[1] * h + h/2)
            ori = stroke[2] * math.pi
            shape_resized.append( (cx,cy,ori) )
        img = sft.draw_shape(shape_resized,img)
        outpath = os.path.join(outdir,'proto%s.jpg'%(cid))
        cv2.imwrite(outpath,img)
                
if __name__=="__main__":
    run('proto','show')
