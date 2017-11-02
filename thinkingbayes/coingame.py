#Exercise 4.1:
#suppose that instead of observing coin toses directly,you measure the outcome using an instrument that is not always
#correct. Specifically, suppose there is a probability y that an actual heads is reported as tails,or actual tails reported
#as heads
#write a class that estimates the bias of a coin given a series of outcomes and the value of y
import os,sys,pdb
import numpy as np
import collections
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
class PMF(object):
    def __init__(self):
        self.N = 100
        self.Y = 0.1 #confidence on observation
        self.Pr = np.zeros((self.N,)) + 1
        self.Pr = self.Pr / self.Pr.sum()
    def likelihood(self,pr,head):
        if head == True:
            return pr * self.Y + (1-pr) * (1-self.Y)
        return (1 - pr) * self.Y + pr * (1 - self.Y)

    def update(self,head):
        likelihood = np.zeros((self.N,))
        for k in range(self.N):
            likelihood[k] = self.likelihood( k * 1.0 / self.N,head)
        self.Pr = self.Pr * likelihood
        self.Pr = self.Pr / self.Pr.sum()
    def show(self):
        X = np.asarray(range(self.N)) * 1.0/ self.N
        Y = self.Pr
        df = pd.DataFrame( {'x':X,'y':Y} )
        sns.jointplot(x='x',y='y',data=df)
        sns.plt.show()
def run():
    heads = []
    for k in range(10):
        heads.append( True )
    for k in range(5):
        heads.append( False )
        pmf = PMF()
    for head in heads:
        pmf.update(head)
    pmf.show()

if __name__=="__main__":
    run()
       

