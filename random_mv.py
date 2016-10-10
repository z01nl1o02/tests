import os,sys,pdb,cPickle
import numpy as np
import random
import argparse
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

def run(indir, outdir, outbatch):
    pathList = load_images(indir)
    random.shuffle(pathList)
    lineList = ['mkdir "%s"'%outdir]
    for index,path in enumerate(pathList):
        line = 'rename "%s" "%s"\\%d.jpg'%(path, outdir, index)
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
    outdir = os.path.splitext(args.outbatch)[0]
    run(args.indir, outdir, args.outbatch) 
