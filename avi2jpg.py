import os,sys,pdb
import cv2
import numpy as np
import argparse

def run(infile, outdir):
    cap = cv2.VideoCapture(infile)
    if not cap.isOpened():
        print "can't open", infile
        return
    fid = 0
    while 1:
        ret, img = cap.read()
        if not ret:
            print 'end with code ',ret
            break
        outfile = os.path.join(outdir,'%.8d.jpg'%fid)
        cv2.imwrite(outfile, img)
        print outfile
        fid += 1

if __name__=="__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('video', help = 'video file')
    ap.add_argument('outdir', help = 'folder to save image')
    args = ap.parse_args()
    try:
        os.makedirs(args.outdir)
    except Exception, e:
        pass
    run(args.video, args.outdir)
