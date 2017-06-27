import os,sys,pdb
import cv2
import numpy as np
import argparse
#REF "Comprehensive Colour Image Normalization"
#    by Graham D. Finlayson
#       Bernt Schiele
#       James L. Crowley

def colorconstance(img):
    size = img.shape[0] * img.shape[1]
    I0 = np.float64(img) / 255.0
    I1 = I0.copy()
    for round in range(100):
        sumGray = np.sum(I1,2)
        sumGray[sumGray < 0.001] = 1.0
        I1[:,:,0] = I1[:,:,0] / sumGray
        I1[:,:,1] = I1[:,:,1] / sumGray
        I1[:,:,2] = I1[:,:,2] / sumGray 
        sumB = I1[:,:,0].sum() * 3
        sumG = I1[:,:,1].sum() * 3
        sumR = I1[:,:,2].sum() * 3
        I1[:,:,0] = I1[:,:,0] * size / sumB  
        I1[:,:,1] = I1[:,:,1] * size / sumG 
        I1[:,:,2] = I1[:,:,2] * size / sumR
        res = np.absolute(I1-I0).sum()
        if res < 0.0001:
            break
        #print 'round ',round,',',res
        I0 = I1.copy()
    ratio = np.sum(img,2)
    ratio[ratio > 255 * 3] = 255 * 3
    I0[:,:,0] = I0[:,:,0] * ratio
    I0[:,:,1] = I0[:,:,1] * ratio
    I0[:,:,2] = I0[:,:,2] * ratio
    I0[ I0 > 255] = 255
    img = np.uint8(I0)
    return img

def run_one_folder(indir,outdir):
    objs = os.listdir(indir)
    for obj in objs:
        sname,ext = os.path.splitext(obj)
        if ext != '.jpg':
            print 'skip ',obj
            continue
        img = cv2.imread(os.path.join(indir, obj), 1)
        img = colorconstance(img)
        cv2.imwrite(os.path.join(outdir, obj), img)
    return

def run(indir, outdir):
    objs = os.listdir(indir)
    for obj in objs:
        fname = os.path.join(indir, obj)
        if not os.path.isdir(fname):
            print 'skip ',fname
            continue
        dstdir = os.path.join(outdir, obj)
        try:
            os.makedirs(dstdir)
        except Exception,e:
            pass
        run_one_folder(fname, dstdir)

if __name__=="__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('inroot', help='input root')
    ap.add_argument('outroot',help='output root')
    args = ap.parse_args()
    run(args.inroot, args.outroot)







