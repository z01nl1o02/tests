import os,sys,pdb
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, recall_score

class SAMPLE(object):
    def __init__(self):
        return
    def load(self,filepath):
        X = []
        Y = []
        with open(filepath,'rb') as f:
            for line in f:
                line = line.strip().split(',')
                x = [ np.float64(k) for k in line[1:]]
                y = np.int64( np.float64(line[0] ) )
                x = np.reshape(  np.asarray(x), (1,-1) )
                X.append(x)
                Y.append(y)
        self.X = X
        self.Y = Y
        return self
    def get_X(self):
        return self.X
    def get_Y(self):
        return self.Y

class CLASSIFICATION(object):
    def __init__(self):
        return
    def CV(self, trainpath, testpath):
        sample = SAMPLE().load(trainpath)
        X = sample.get_X()
        Y = sample.get_Y()
           
        tsample = SAMPLE().load(testpath)
        testX = tsample.get_X()
        testY = tsample.get_Y()
        testX = np.vstack(testX)
        testY = np.asarray(testY)
        
        recalls = []
        precisions = []
        testrecalls = []
        testprecisions = []
        kf = KFold(n_splits=3)
        for train,valid in kf.split(X):
            trainX = []
            trainY = []
            validX = []
            validY = []
          
            for idx in train:
                trainX.append(  X[idx] )
                trainY.append( Y[idx] )
            for idx in valid:
                validX.append( X[idx] )
                validY.append( Y[idx] )
            trainX = np.vstack( trainX )
            trainY = np.asarray( trainY )
            validX = np.vstack( validX )
            validY = np.asarray( validY )
            clf = LinearSVC(random_state=0).fit(trainX,trainY)
            C = clf.predict(validX)
            recalls.append(  recall_score(validY,C,average='micro') )
            precisions.append( accuracy_score(validY,C) )
            C = clf.predict(testX)
            
            testrecalls.append(  recall_score(testY,C,average='micro') )
            testprecisions.append( accuracy_score(testY,C) )
        for r, p,tr,tp in zip(recalls, precisions, testrecalls, testprecisions):
            print 'valid: recall %f precsion %f, test: recall %f precisions %f'%(r,p,tr,tp)
        print 'valid: ',np.asarray(recalls).mean(), np.asarray(precisions).mean()
        print 'test: ',np.asarray(testrecalls).mean(), np.asarray(testprecisions).mean()

if __name__=="__main__":
    clf = CLASSIFICATION().CV('train.txt','test.txt')