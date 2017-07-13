import pandas as pd
import numpy as np
import os,sys,pdb
from collections import Counter
import argparse
class CVT_CATEG:
    def __init__(self,df,datadir):
        self._df = df
        self._datadir = datadir
        self._categ2idx = {}
        try:
            os.makedirs(datadir)
        except Exception,e:
            pass
        self.gen_dict()
    def gen_dict(self):
        df = self._df
        outdir = self._datadir
        feats = df.dtypes[ df.dtypes == 'object' ].index
        for feat in feats:
            vals = list(df[feat])
            vdict = dict(Counter(vals))
            vdict = sorted(vdict.iteritems(), key = lambda x:x[1], reverse=True)
            lines = []
            for val,num in vdict:
                lines.append('%s,%d'%(val,num))
            with open(os.path.join(outdir,feat + '.txt'), 'wb') as f:
                f.writelines('\r\n'.join(lines))
        return
    def load_one_feat(self,feat):
        path = os.path.join(self._datadir, feat + '.txt')
        if not os.path.exists(path):
            return False
        self._categ2idx = {}
        lut = self._categ2idx
        with open(path, 'rb') as f:
            idx = 0
            for line in f:
                line = line.strip()
                if line == '':
                    continue
                categ,num = line.split(',')
                lut[categ] = idx
                idx += 1
        return True
    def lookup(self,categ):
        if categ in self._categ2idx:
            return self._categ2idx[categ]
        return None


if __name__=="__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('infile', help='input csv')
    ap.add_argument('outdir', help='output dir')
    args = ap.parse_args()
    df = pd.read_csv(args.infile)
    cvtcateg = CVT_CATEG(df,args.outdir)
    cvtcateg.load_one_feat('BsmtFinType2')
    print "NA", cvtcateg.lookup('nan')
    print "ALQ", cvtcateg.lookup('ALQ')



