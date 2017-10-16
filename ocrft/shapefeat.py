import os,sys,pdb
import cv2
import numpy as np
from toshape import SHAPE_FEAT

def save_feat(shapes,cid,outfile):
    lines = []
    for shape in shapes:
        data = [cid]
        for vec in shape:
            cx,cy,ori = vec
            data.append( '%f,%f,%f'%(cx,cy,ori))
        data.append(outfile)
        lines.append( '|'.join(data) )
    with open(outfile,'ab+') as f:
        f.writelines('\r\n'.join(lines))

def run_one_class(cid,indir,outdir):
    try:
        os.makedirs(outdir)
    except Exception,e:
        print e
    shapes = []
    fg = SHAPE_FEAT(False,8,8)
    for name in os.listdir(indir):
        sname,ext = os.path.splitext(name)
        if ext != '.jpg' and ext != '.bmp':
            continue
        bw = cv2.imread(os.path.join(indir,name),0)
        shape = fg.simulate_contour(bw)
        shapes.append(shape) 
    save_feat(shapes,cid,os.path.join(outdir,cid+'.txt'))

def run_all(indir,outdir):
    for cid in os.listdir(indir):
        idir = os.path.join(indir,cid)
        run_one_class(cid,idir,outdir)

if __name__=="__main__":
    run_all('thin','feats')

