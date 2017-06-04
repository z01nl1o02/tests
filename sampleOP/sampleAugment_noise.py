import os,sys,pdb
import cv2
import argparse
import numpy as np
import multiprocessing as mp
import random
import math


class SALT_PEPPER:
    def __init__(self,SNR = 0.995):
        self.SNR = SNR
        return
    def run(self,gray):
        h,w = gray.shape
        NP = np.int64((w * h) * (1 - self.SNR))
        for k in range(NP):
            y = random.randint(0, h - 1)
            x = random.randint(0, w - 1)
            g = random.uniform(0,1)
            if g > 0.5:
                g = 255
            else:
                g = 0
            gray[y,x] = g
        return gray

class GAUSSIAN:
    def __init__(self, mean = 0.0, sigma = 0.05):
        sigma = sigma * 255
        mean = mean * 255
        sigma2 = sigma * sigma * 2
        sigmaPI2 = math.sqrt( math.pi * 2) * sigma
        dist = [ 0 for k in range(-255,256)]
        for value in range(-255,256):
            pos = value + 255
            dist[pos] = math.exp(-((value - mean) ** 2)/ sigma2) / sigmaPI2
        self.pdf = []
        self.pdf.append( dist[0] )
        for value in range(len(dist)):
            self.pdf.append(  self.pdf[-1] + dist[value] )
    
    def get_value(self):
        prob = random.uniform(0,1)
        value = 0
        for pos in range(1,len(self.pdf)):
            if prob >= self.pdf[pos - 1] and prob <= self.pdf[pos]:
                value = pos - 255
                break
        return value

    def run(self, gray):
        h,w = gray.shape
        for y in range(h):
            for x in range(w):
                val = gray[y,x] + self.get_value()
                if val < 0:
                    val = 0
                elif val > 255:
                    val = 255
                gray[y,x] = val
        return gray

def run_one_class(args):
    indir, outdir, noisetype = args
    try:
        os.makedirs(outdir)
    except Exception,e:
        pass
    if noisetype == 'gaussian':
        noise = GAUSSIAN()
    elif noisetype == 'saltpepper':
        noise = SALT_PEPPER()
    else:
        print 'unk noise type ',noisetype
        return 
    jpgs = os.listdir(indir)
    for jpg in jpgs:
        sname,ext = os.path.splitext(jpg)
        if '.jpg' != ext:
            continue
        img = cv2.imread(os.path.join(indir,jpg),0)
        img = noise.run(img)
        cv2.imwrite( os.path.join(outdir, sname + ",noise_%s.jpg"%noisetype), img)
    return

def run(indir,outdir,noisetype,cpu):
    params = []
    objs = os.listdir(indir)
    for obj in objs:
        fname = os.path.join(indir, obj)
        if not os.path.isdir(fname):
            continue
        params.append( (fname, os.path.join(outdir,obj), noisetype) )
    pool = mp.Pool(cpu)
    pool.map(run_one_class, params)
    pool.close()
    pool.join()
    return

def run_S(indir,outdir,noisetype,cpu):
    params = []
    objs = os.listdir(indir)
    for obj in objs:
        fname = os.path.join(indir, obj)
        if not os.path.isdir(fname):
            continue
        run_one_class((fname, os.path.join(outdir,obj), noisetype));
    return


if __name__=="__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('indir',help='input dir')
    ap.add_argument('outdir',help='output dir')
    ap.add_argument('type',help='noise type (saltpepper,gaussian)',default='saltpepper')
    ap.add_argument('-cpu',help='thread number',default=2, type=np.int64)
    args = ap.parse_args()
    run_S(args.indir, args.outdir, args.type, args.cpu)




