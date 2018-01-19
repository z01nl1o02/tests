import os,sys,pdb
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, recall_score
X = []
Y = []
with open('results.csv','rb') as f:
    for line in f:
        line = line.strip().split(',')
        x = [ np.float64(k) for k in line[1:]]
        y = np.int64( np.float64(line[0] ) )
        x = np.reshape(  np.asarray(x), (1,-1) )
        X.append(x)
        Y.append(y)

recalls = []
precisions = []
kf = KFold(n_splits=3)
for train,test in kf.split(X):
    trainX = []
    trainY = []
    testX = []
    testY = []
    for idx in train:
        trainX.append(  X[idx] )
        trainY.append( Y[idx] )
    for idx in test:
        testX.append( X[idx] )
        testY.append( Y[idx] )
    trainX = np.vstack( trainX )
    trainY = np.asarray( trainY )
    testX = np.vstack( testX )
    testY = np.asarray( testY )
    C = LinearSVC(random_state=0).fit(trainX,trainY).predict(testX)
    recalls.append(  recall_score(testY,C) )
    precisions.append( accuracy_score(testY,C) )

for r, p in zip(recalls, precisions):
    print 'recall %f precsion %f'%(r,p)
print np.asarray(recalls).mean(), np.asarray(precisions).mean()

