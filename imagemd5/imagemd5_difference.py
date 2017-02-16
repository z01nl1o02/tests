import os,sys,pdb,cPickle
import pandas as pd
import argparse
import numpy as np

def run(firstFile, secondFile,outcsv):
    with open(firstFile, 'rb') as f:
        dictA = cPickle.load(f)
    with open(secondFile, 'rb') as f:
        dictB = cPickle.load(f)
    keyA = set(dictA.keys())
    keyB = set(dictB.keys())
    keys = list(keyA - keyB)
    print 'difference(A-B) total %d'%len(keys)
    dictC = dictA
    line_list = []
    for key in keys:
        files = dictC[key]
        line_list.extend(files)
    line_list = set(list(line_list))
    line = '\r\n'.join(line_list)
    with open(outcsv,'wb') as f:
        f.writelines(line)

if __name__=="__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('firstpkl',help='first pkl')
    ap.add_argument('secondpkl',help='second pkl')
    ap.add_argument('outcsv',help='output csv')
    args = ap.parse_args()
    run(args.firstpkl, args.secondpkl, args.outcsv)

    

