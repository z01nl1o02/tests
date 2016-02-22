"""
clf = OneClassSVM(kernel='rbf',gamma=gamma_value,nu=nu_value)
the smaller gamma is, the smoother contour is
the smaller nu is, the more samples included in countour
problem with OneClassSVM(): hard to set parameter


clf = EllipticEnvelope(contamination=0.1)
contamination: proportion of outliers in data set
"""
print(__doc__)

import os,sys,pdb,pickle
from sklearn.covariance import EllipticEnvelope
from sklearn.svm import OneClassSVM
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def load_one_class_feature(filepath):
    featlist = []
    with open(filepath, 'r') as f:
        for line in f:
            feat = line.strip().strip(',;').split(',')
            feat = [float(k) for k in feat]
            featlist.append(feat)
    return featlist

def norm_data(X):
    X = np.array(X)
    m0 = X.min(0)
    m1 = X.max(0)
    m = m1 - m0
    idx = m < 0.001
    m[idx] = 1
    m0 = np.tile(m0, (X.shape[0],1))
    m = np.tile(m, (X.shape[0],1))
    X = (X - m0 ) / m
    X[:,idx] = 0
    clf = PCA(2)
    X = clf.fit_transform(X)
    return X


def show(samplepath):
    paths = []
    sname = os.path.splitext(samplepath)[0]
    print sname
    with open(sname+"_path.txt", 'r') as f:
        for line in f:
            paths.append(line.strip())
    X = load_one_class_feature(samplepath)
    X = norm_data(X)
    #clf = OneClassSVM(kernel='rbf',gamma=0.01,nu=0.098)
    clf = EllipticEnvelope(contamination=0.05)
    clf.fit(X)
    Y = clf.predict(X)
    DY = clf.decision_function(X)
    for k in range(len(Y)):
        if Y[k] < 0: #abnormality is positive
            print k + 1, ',', DY[k], ',',paths[k]
    err = np.sum( [ y < 0 for y in Y] )
    print '%d/%d'%(err, len(Y))

    x1,y1 = np.meshgrid(np.linspace(-20,20,400), np.linspace(-20,20,400))
    z1 = clf.decision_function(np.c_[x1.ravel(), y1.ravel()])
    z1 = z1.reshape(x1.shape)
    legend = {}
    legend['test'] = plt.contour(x1,y1,z1, levels=[0], linewidths=2,color='r')
    plt.scatter(X[:,0], X[:,1], color='black')

    values_list = list(legend.values())
    keys_list = list(legend.keys())
    plt.legend([values_list[0].collections[0]],[keys_list[0]])
    plt.show()

if __name__=="__main__":
    show(sys.argv[1])







