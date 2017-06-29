import os,sys,pdb
import numpy as np
from collections import defaultdict
import argparse

class EVAL_CLASS:
    def __init__(self):
        self._data = defaultdict(dict)
        self._names = []
    def add(self,Y,C):
        Y = str(Y)
        C = str(C)
        #Y: label name  (str only)
        #C: clf ouptut name (str only)
        if Y not in self._data.keys():
            self._data[Y] = {}
        if C not in self._data[Y]:
            self._data[Y][C] = 0
        self._data[Y][C] += 1
        return
    def format(self):
        if self._data is None:
            return
        self._classnames = []
        for Y in self._data.keys():
            self._names.append(Y)
        self._names = sorted( list( set(self._names) ) )
        for Y in self._names:
            for C in self._names:
                if C in self._data[Y].keys():
                    continue
                self._data[Y][C] =  0
        return
    def write(self,filepath):
        self.format()
        lines = []
        lines.append(','.join(self._names))
        for Y in self._names:
            line = []
            for C in self._names:
                line.append('%d'%self._data[Y][C])
            lines.append( ','.join(line) )
        with open(filepath, 'wb') as f:
            f.writelines('\r\n'.join(lines))
        return
    def read(self,filepath):
        self._data = defaultdict(dict)
        self._names = []
        with open(filepath, 'rb') as f:
            line = f.readline().strip()
            self._names = line.split(',')
            for Y in self._names:
                self._data[Y] = {}
                for C in self._names:
                    self._data[Y][C] = 0
            lineNO = 0
            for line in f:
                Y = self._names[lineNO]
                lineNO += 1
                data = [np.int64(X) for X in line.strip().split(',')]
                for C, num in zip(self._names,data):
                    self._data[Y][C] = num
        return
    def cvt2mat(self):
        mat = np.zeros((len(self._names), len(self._names)))
        for row,Y in enumerate(self._names):
            for col,C in enumerate(self._names):
                mat[row,col] = self._data[Y][C]
        return mat

    def recalling(self):
        res = []
        mat = self.cvt2mat()
        for row,Y in enumerate(self._names):
            TPFN = mat[row,:].sum()
            if TPFN < 1:
                TPFN = 1
            TP = mat[row,row]
            res.append( TP * 1.0 / TPFN )
        return res

    def precision(self):
        res = []
        mat = self.cvt2mat()
        for col,Y in enumerate(self._names):
            TPFP = mat[:,col].sum()
            if TPFP < 1:
                TPFP = 1
            TP = mat[col,col]
            res.append( TP * 1.0 / TPFP )
        return res
    def F1(self,beta = 1.0):
        rec = self.recalling()
        pre = self.precision()
        res = []
        for r,p in zip(rec,pre):
            res.append( (1 + beta**2) * (r * p) / (r + (beta**2)*p) )
        return res
    def show(self):
        R = self.recalling()
        P = self.precision()
        F1 = self.F1()
        res = []
        for n, r, p, f1 in zip(self._names, R,P,F1):
            res.append( (n, r,p,f1) )
        res = sorted(res, key = lambda X:X[-1])
        print 'name,recalling,precision,F1'
        for n, r, p, f1 in res:
            print '%20s,%8.3f,%8.3f,%8.3f'%(n,r,p,f1)
if __name__=="__main__":
    stat = EVAL_CLASS()
    stat.read(sys.argv[1])
    stat.show()

