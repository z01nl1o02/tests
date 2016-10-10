import os,sys,pdb,cPickle
import numpy as np
import random
import argparse
import time
def load_images(indir):
    pathList = []
    for rdir, pdirs, names in os.walk(indir):
        for name in names:
            sname,ext = os.path.splitext(name)
            if '.jpg' != ext:
                print 'skip %s'%name
                continue
            pathList.append( os.path.join(rdir,name) )
    return pathList

def run(indir, outbatch):
    timestamp = time.time()
    pathList = load_images(indir)
    random.shuffle(pathList)
    lineList = []
    for index,path in enumerate(pathList):
        line = 'rename "%s" "%d_%d.jpg'%(path, timestamp, index)
        lineList.append(line)
    lines = '\r\n'.join(lineList)
    with open(outbatch,'wb') as f:
        f.writelines(lines)
    return

if __name__=="__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('indir',help='image root folder')
    ap.add_argument('outbatch',help='outupt batch file')
    args = ap.parse_args()
    run(args.indir, args.outbatch) 
