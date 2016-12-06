import os,sys,pdb
import glob
import argparse
import numpy as np

def split_one_class(indir, trainOutdir, testOutdir, minNumToSplit, trainRatio):
    className = indir.strip(os.path.sep).split(os.path.sep)[-1]
    traindir = os.path.join(trainOutdir,className)
    testdir = os.path.join(testOutdir,className)
    lines = ['mkdir "%s"'%traindir]
    lines.append('mkdir "%s"'%testdir)
    jpgs = glob.glob(os.path.join(indir,"*.jpg"))
    if len(jpgs) == 1:
        jpg = jpgs[0]
        lines.append('copy "%s" "%s"'%(jpg, traindir))
        lines.append('copy "%s" "%s"'%(jpg, testdir))
        return lines

    if len(jpgs) < minNumToSplit:
        numTrain = len(jpgs)/2
        if numTrain + 1 < len(jpgs):
            numTrain += 1 #more for train
    else:
        numTrain = np.int64(len(jpgs) * trainRatio )

    for idx, jpg in enumerate (jpgs):
        if idx < numTrain:
            lines.append('copy "%s" "%s"'%(jpg, traindir))
        else:
            lines.append('copy "%s" "%s"'%(jpg, testdir))
    return lines

def run(indir, outdir, minNumToSplit, trainRatio):
    batID = 0
    lines = []
    batFiles = []
    traindir = os.path.join(outdir,'train')
    testdir = os.path.join(outdir,'test')
    try:
        os.makedirs(traindir)
        os.makedirs(testdir)
    except Exception,e:
        pass
    objs = os.listdir(indir)
    for obj in objs:
        fname = os.path.join(indir, obj)
        if not os.path.isdir(fname):
            continue
        lines.extend( split_one_class(fname, traindir,testdir,minNumToSplit, trainRatio) )
        if len(lines) > 100000:
            outfile = 'split%d.bat'%batID
            batFiles.append("start %s"%outfile)
            batID += 1
            with open(outfile, "wb") as f:
                f.writelines('\r\n'.join(lines))
            lines = [] 
    if len(lines) > 0:
        outfile = 'split%d.bat'%batID
        batFiles.append('start %s'%outfile)
        with open(outfile, "wb") as f:
            f.writelines('\r\n'.join(lines))
    allbat = '_callall.bat'
    with open(allbat, 'wb') as f:
        f.writelines( '\r\n'.join(batFiles) )
    print 'call %s to finish work!'%allbat
    return


if __name__=="__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('indir',help='input dir')
    ap.add_argument('outdir',help='output dir')
    ap.add_argument('-minNumToSplit',help='minimum to do split',type=np.int64,default=10) 
    ap.add_argument('-trainRatio',help='ratio of trainset in all set', type=np.float64, default=0.8)
    args = ap.parse_args()
    run(args.indir, args.outdir, args.minNumToSplit, args.trainRatio)









