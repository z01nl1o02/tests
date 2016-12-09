import os,sys,pdb
import cv2
from multiprocessing import Pool
import argparse
import numpy as np

def gen_block_from_one_class(args):
    indir,blockNumX,blockNumY,classname,outdir = args
    objs = os.listdir(indir)
    for obj in objs:
        fname = os.path.join(indir,obj)
        img = cv2.imread(fname,1)
        blockw = img.shape[1] / blockNumX
        blockh = img.shape[0] / blockNumY
        blockID = -1
        for by in range(blockNumY):
            for bx in range(blockNumX):
                blockID += 1
                x0 = bx * blockw
                y0 = by * blockh
                x1 = x0 + blockw
                y1 = y0 + blockh
                if x1 > img.shape[1] or y1 > img.shape[0]:
                    continue
                outfile = os.path.join(outdir,'part%d'%blockID)
                outfile = os.path.join(outfile,classname)
                outfile = os.path.join(outfile,obj)
                cv2.imwrite(outfile, img[y0:y1,x0:x1,:])
    return

def run(indir, outdir, blockNumX, blockNumY, cpu):
    partdirs = []
    for k in range(blockNumX * blockNumY):
        partdirs.append( os.path.join(outdir, 'part%d'%k) )
    try:
        for k in range(blockNumX * blockNumY):
            os.makedirs( partdirs[k] )
    except Exception,e:
        pass
    params = []
    objs = os.listdir(indir)
    for classname in objs:
        try:
            map( lambda X: os.makedirs( os.path.join(X,classname) ), partdirs)
        except Exception,e:
            pass
        params.append( (os.path.join(indir,classname), blockNumX, blockNumY, classname, outdir ) )
    pool = Pool(cpu)
    pool.map( gen_block_from_one_class, params)
    pool.close()
    pool.join()
    return


if __name__=="__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('indir',help='input dir with sub-folders')
    ap.add_argument('outdir',help='output  dir')
    ap.add_argument('-cpu',help='thread number',type=np.int64,default=3)
    ap.add_argument('numX',help='block number along x-axis',type = np.int64)
    ap.add_argument('numY',help='block number along y-axis', type = np.int64)
    args = ap.parse_args()
    run(args.indir, args.outdir, args.numX, args.numY, args.cpu)

