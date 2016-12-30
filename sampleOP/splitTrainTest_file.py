import os,sys,pdb
import glob
import argparse
import numpy as np

def split_one_class(infile, trainOutdir, testOutdir, minNumToSplit, trainRatio):
    className = infile.strip(os.path.sep).split(os.path.sep)[-1]
    className = os.path.splitext(infile)[0].split(os.path.sep)[-1]
    traindir = os.path.join(trainOutdir,className)
    testdir = os.path.join(testOutdir,className)
    lines = []
    with open(infile,'rb') as f:
        for line in f:
            line = line.strip()
            if line == "":
                continue
            lines.append(line)
    if len(lines) < minNumToSplit:
        numTrain = len(lines)/2
        if numTrain + 1 < len(lines):
            numTrain += 1 #more for train
    else:
        numTrain = np.int64(len(lines) * trainRatio )

    lineTrain = []
    lineTest = []
    for idx, line in enumerate (lines):
        if idx < numTrain:
            lineTrain.append(line)
        else:
            lineTest.append(line)
    with open(os.path.join(trainOutdir,className+'.txt'), 'wb') as f:
        f.writelines('\r\n'.join(lineTrain))
    with open(os.path.join(testOutdir,className+'.txt'), 'wb') as f:
        f.writelines('\r\n'.join(lineTest))
    return

def run(indir, outdir, minNumToSplit, trainRatio):
    batID = 0
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
        split_one_class(fname, traindir,testdir,minNumToSplit, trainRatio)
    return


if __name__=="__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('indir',help='input dir')
    ap.add_argument('outdir',help='output dir')
    ap.add_argument('-minNumToSplit',help='minimum to do split',type=np.int64,default=10) 
    ap.add_argument('-trainRatio',help='ratio of trainset in all set', type=np.float64, default=0.8)
    args = ap.parse_args()
    run(args.indir, args.outdir, args.minNumToSplit, args.trainRatio)



