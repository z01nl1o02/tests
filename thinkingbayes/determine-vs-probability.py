"""
there are N balls in one bag. one fetch with ratio f each time.
for first try, one gets 30 balls (without return)
for second try, one gets 20 balls
then what the value of N and f 

surely one may solve it by equation like this
N * f = 30
N * (1-f) * f = 20
so
N = 90
f = 1/3

is it possible to sovle it by probability?
if you put (90,1/3) in hypos, you may get it with max-post-probability
but without (90,1/3), you put (90,0.333) in hypos, you may get another result
"""

import os,sys,pdb
import numpy as np
from scipy.stats import binom
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

class DISTRIBUTION(object):
    def __init__(self,hypos):
        self.hypos = hypos
        self.pr = np.zeros( (1, len(hypos))) + 1
        self.pr = self.pr / self.pr.sum()
    def likelihood(self,hypo,data):
        idx,k = data
        N,f = hypo
        N = N * ((1-f) ** (idx-1)) #1-based
        if N < k:
            return 0
        val = binom.pmf(k,N,f)
        return val
    def update(self,data):
        for k,hypo in enumerate(self.hypos):
            likelihood = self.likelihood(hypo,data)
            self.pr[0,k] *= likelihood
        self.pr = self.pr / self.pr.sum()
    def show(self):
        pos = np.argsort(self.pr)[0][-20:]
        for k in pos:
            print self.hypos[k],self.pr[0,k]
        pos = np.argmax(self.pr)
        print 'max',self.hypos[pos],'pr=',self.pr[0,pos]
        X = []
        for idx,hypo in enumerate(self.hypos):
            N,f = hypo
            X.append(idx)
        Y = self.pr.tolist()[0]
        df = pd.DataFrame({'x':X,'y':Y})
        sns.jointplot(x='x',y='y',data=df)
        sns.plt.show()

hypos = []
for N in range(30,200,10):
    for f in range(1,1000):
        hypos.append( (N,f/1000.0) )
"""
for N in range(30,200,10):
    for f in range(1,1000):
        hypos.append( (N,1.0/f) )
"""
dist = DISTRIBUTION(hypos)
datas = [30,20]
for idx,data in enumerate(datas):
    dist.update( (idx + 1,data) )
dist.show()



