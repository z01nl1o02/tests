import os,sys,pdb
import cv2
import argparse
import numpy as np
import multiprocessing as mp

def run_one_class(args):
    indir, outdir, fliptype = args
    try:
        os.makedirs(outdir)
    except Exception,e:
        pass
    jpgs = os.listdir(indir)
    for jpg in jpgs:
        sname,ext = os.path.splitext(jpg)
        if '.jpg' != ext:
            continue
        img = cv2.imread(os.path.join(indir,jpg))
        if fliptype == 'ver':
            flipImg = cv2.flip(img,0)
        elif fliptype == 'hor':
            flipImg = cv2.flip(img,1)
        else:
            continue
        cv2.imwrite( os.path.join(outdir, sname + ",%s.jpg"%fliptype), flipImg)
    return

def run(indir,outdir,fliptype,cpu):
    params = []
    objs = os.listdir(indir)
    for obj in objs:
        fname = os.path.join(indir, obj)
        if not os.path.isdir(fname):
            continue
        params.append( (fname, os.path.join(outdir,obj), fliptype) )
    pool = mp.Pool(cpu)
    pool.map(run_one_class, params)
    pool.close()
    pool.join()
    return

if __name__=="__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('indir',help='input dir')
    ap.add_argument('outdir',help='output dir')
    ap.add_argument('fliptype',help='flip type (ver,hor)',default='hor')
    ap.add_argument('-cpu',help='thread number',default=2, type=np.int64)
    args = ap.parse_args()
    run(args.indir, args.outdir, args.fliptype, args.cpu)




