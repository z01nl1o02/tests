import os,sys,pdb,cv2,pickle
import numpy as np
import feat_lbp
import feat_hog
from sklearn.svm import SVC

class CLF_SVM(object):
    def __init__(self, ft, pf):
        self.fh = None
        self.clf = None
        self.minmaxrange = []
        self.clfpath = 'CLF_SVM_'+ft+'_'+pf+'.dat'
        if 0 == cmp(ft, 'lbp'):
            print "ft : LBP"
            self.fh = feat_lbp.FEAT_LBP()
        elif 0 == cmp(ft, 'hog'):
            print "ft : HOG"
            self.fh = feat_hog.FEAT_HOG()
        else:
            print 'unknown feature type'

    def normalization(self, samples):
        if len(self.minmaxrange) == 0:
            m0 = np.reshape(samples.min(0),(1,-1))
            m1 = np.reshape(samples.max(0),(1,-1))
            ran = (m1 - m0) + 0.0001
            self.minmaxrange = [m0,m1,ran]

        m0 = self.minmaxrange[0]
        ran = self.minmaxrange[2]

        num = samples.shape[0]
        m0 = np.tile(m0, (num,1))
        ran = np.tile(ran, (num,1))

        samples = (samples - m0) / ran
        return samples

    def get_samples(self, folderpath,count):
        if self.fh is None:
            print 'null feature handle'
            return (None,None)
        fvs, paths = self.fh.folder_mode(folderpath,count)
        fvs = np.array(fvs)
        return (fvs,paths)

    def predict(self, folderpath):
        if self.clf is None:
            with open(self.clfpath, 'rb') as f:
                self.clf, self.minmaxrange = pickle.load(f)
        if self.clf is None:
            print 'clf is null'
            return
        tests,paths = self.get_samples(folderpath,-1)
        tests = self.normalization(tests)
        print 'test ', tests.shape
        prds = self.clf.predict(tests)
        posnum = 0
        pos = ""
        neg = ""
        path2prd = {}
        for prd, path in zip(prds, paths):
            if prd == 1:
                pos += path + '\n'
                posnum += 1
                path2prd[path] = 1
            else:
                neg += path + '\n'
                path2prd[path] = 0
        with open('pos.txt', 'w') as f:
            f.writelines(pos)
        with open('neg.txt', 'w') as f:
            f.writelines(neg)
        print 'predict : ', len(prds), ',', posnum * 1.0 / len(prds)
        return path2prd

    def train(self, dataset, count):
        posinfo = self.get_samples(dataset + '/pos',count)
        print 'pos ', posinfo[0].shape, ' ',
        neginfo = self.get_samples(dataset + '/neg',count)
        print 'neg ', neginfo[0].shape
        posnum = posinfo[0].shape[0]
        negnum = neginfo[0].shape[0]
        samples = np.vstack((posinfo[0], neginfo[0]))
        samples = self.normalization(samples)
        paths = posinfo[1].extend(neginfo[1])
        labels = [1 for k in range(posnum)] + [0 for k in range(negnum)]
        self.clf = SVC(C=1.0,kernel='linear',verbose=False).fit(samples, labels)
        prds = self.clf.predict(samples)
        TP = 0
        TN = 0
        for k in range(prds.shape[0]):
            if prds[k] == 1 and labels[k] == 1:
                TP += 1
            if prds[k] == 0 and labels[k] == 0:
                TN += 1
        print 'TP :', posnum ,',',TP * 1.0/posnum, ' ',
        print 'TN :', negnum ,',',TN * 1.0/negnum
        with open(self.clfpath, 'wb') as f:
            pickle.dump((self.clf,self.minmaxrange), f)
        return 

def do_train_bagging(dataset, ft,modelnum):
    for k in range(modelnum):
        clf = CLF_SVM(ft,str(k))
        clf.train(dataset,1000)

def do_test(folderpath, ft, modelnum):
    path2prd = {} 
    for k in range(modelnum):
        clf = CLF_SVM(ft,str(k))
        p2p = clf.predict(folderpath) 
        for path in p2p.keys():
            if path not in path2prd:
                path2prd[path] = p2p[path]
            else:
                path2prd[path] += p2p[path]
    posnum = 0
    thresh = modelnum / 2.0
    for path in path2prd.keys():
        if path2prd[path] > thresh:
            posnum += 1
    print 'predict all : ', len(path2prd) , ',' , posnum * 1.0 / len(path2prd)

if __name__=="__main__":
    if len(sys.argv) == 4 and 0 == cmp(sys.argv[1],'-train'):
        ft = sys.argv[2]
        modelnum = np.int32(sys.argv[3])
        with open('config.txt', 'r') as f:
            dataset = f.readline().strip()
        do_train_bagging(dataset, ft,modelnum)
    elif len(sys.argv) == 5 and 0 == cmp(sys.argv[1],'-test'):
        ft = sys.argv[2]
        modelnum = np.int32(sys.argv[3])
        folderpath = sys.argv[4]
        do_test(folderpath, ft,modelnum)
    else:
        print 'unknown option'
