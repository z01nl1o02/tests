import os,sys,pdb
import numpy as np
import math
import copy
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

"""This file contains a partial solution to a problem from
MacKay, "Information Theory, Inference, and Learning Algorithms."

    Unstable particles are emitted from a source and decay at a
distance $x$, a real number that has an exponential probability
distribution with [parameter] $\lambda$.  Decay events can only be
observed if they occur in a window extending from $x=1$ cm to $x=20$
cm.  $N$ decays are observed at locations $\{ 1.5, 2, 3, 4, 5, 12 \}$
cm.  What is $\lambda$?

"""
class EXPPDF(object):
    def __init__(self,lams):
        self._lams = copy.deepcopy(lams)
        self._pr = np.zeros((len(lams),)) + 1
        self._pr = self._pr / self._pr.sum()
    def likelihood(self,lam,data):
        factor = math.exp(-1 * lam) - math.exp(-20 * lam) 
        """
        why "factor"? the following is one explaination but still strange???
        1. according to text, it should be one condition probability
           Pr(x | 1 < x < 20) = Pr(x) * Pr(1<x<20 | x) / Pr(1 < x < 20)
           Pr(1<x<20|x) ????
        2.PMF of exp distribution  Pr(t < x) = 1 - exp(-lam * x)
        3.so ....
        
        objective:
        with factor the mean of lambda is 3 
        without factor the mean of lambda is 2.5
        
        a = math.exp(-1 * 2.5) - math.exp(-20 * 2.5) = 0.08
        b = math.exp(-1 * 3) - math.exp(-20 * 2) = 0.04
        a > b so lam = 2.5 more x fails in [1,20]
        """
        return lam * math.exp(-1 * lam * data) / factor
    def update(self,data):
        for k,lam in enumerate(self._lams):
            pr = self.likelihood(lam,data)
            self._pr[k] *= pr
        self._pr /= self._pr.sum()
    def show(self):
        Y = np.reshape(self._pr,(1,-1)).tolist()[0]
        X = self._lams
        df = pd.DataFrame({'x':X,'y':Y})
        sns.jointplot(x='x',y='y',data=df)
        Y = np.asarray(Y)
        X = np.asarray(X)
        mean = (X*Y).sum()
        sns.plt.title('mean %f'%mean)
        sns.plt.show()

lams = np.linspace(1,100,1000) * 1.0 / 100
pdf = EXPPDF(lams)
datas = [1.5,2,3,4,5,12]
for data in datas:
    pdf.update(data)
pdf.show()

#as datas contains more observed x [0,20] the mean of lam decreases!!!!!!!!!!




