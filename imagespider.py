import sys,os,pdb,pickle,gc
import cv2
import numpy as np
from sklearn.cluster import MiniBatchKMeans,KMeans
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
class IMAGE_SPIDER:
    def extract_raw_feature_single(self, imgpath):
        img = cv2.imread(imgpath,0)
        stdw,stdh = (512,256)
        img = cv2.resize(img, (stdw,stdh))
        dx = cv2.Sobel(img, cv2.CV_32F, 0,1) 
        dy = cv2.Sobel(img, cv2.CV_32F, 1,0)
        grad = np.sqrt( dx**2 + dy**2) 
        feats = []
        for y in range(0, grad.shape[0], 8):
            for x in range(0, grad.shape[1], 8):
                f = grad[y:y+8,x:x+8]
                f = np.reshape(f,(1,-1))
                feats.append(f)
        return feats
        
           
    def extract_raw_feature(self,imgdir,samplepath, pcapath):
        #1--feature extraction
        feats = []
        for rdirs, pdirs, names in os.walk(imgdir):
            for name in names:
                sname,ext = os.path.splitext(name)
                if 0 == cmp(ext, '.jpg'):
                    fname = os.path.join(rdirs, name)
                    feat = self.extract_raw_feature_single(fname)
                    feats.extend(feat)
                    print sname

        #2--list2matrix
        samplenum = len(feats)
        featsize = feats[0].shape[1]
        samples = np.zeros((samplenum, featsize))
        for k in range(samplenum):
            samples[k,:] = feats[k]
        del feats
        gc.collect()


        #3--pca
        pca = PCA(0.98)
        samples = pca.fit_transform(samples)


        #4--save model
        with open(samplepath,'w') as f:
            pickle.dump(samples,f)

        with open(pcapath, 'w') as f:
            pickle.dump(pca, f)

    def create_bow_dict(self, samplepath, pcapath, dictpath):
        with open(samplepath,'r') as f:
            samples = pickle.load(f)
        with open(pcapath,'r') as f:
            pca = pickle.load(f)

        clusternum = 50
        print 'run kmeans with cluster = ', clusternum
        bowdict = MiniBatchKMeans(clusternum,verbose=True).fit(samples)

        with open(dictpath, 'w') as f:
            pickle.dump(bowdict, f)
        print 'dict is done!'

    def create_bow_feature_single(self, imgpath, pca, bowdict):
        #1--raw feature
        feats = self.extract_raw_feature_single(imgpath)
        samples = np.zeros((len(feats), feats[0].shape[1]))
        for k in range(samples.shape[0]):
            samples[k,:] = feats[k]
        #2--pca
        samples = pca.transform(samples)
        #3--kmeans
        labels = bowdict.predict(samples)
        feat = np.zeros((1, bowdict.cluster_centers_.shape[0]))
        for k in range(labels.shape[0]):
            feat[0,labels[k]] += 1
        return feat

    def create_bow_feature(self, imgdir, pca, bowdict):
        feats = []
        labels = []
        imgpaths = []
        for rdir, pdir, names in os.walk(imgdir):
            for name in names:
                sname,ext = os.path.splitext(name)
                if 0 == cmp(ext, '.jpg'):
                    fname = os.path.join(rdir,name)
                    feat = self.create_bow_feature_single(fname, pca, bowdict)
                    feats.append(feat)
                    imgpaths.append(fname)
                    if 0 == cmp(name[6:9], 'pos'): #pos / neg
                        labels.append(1)
                    else:
                        labels.append(0)
        return (labels,feats, imgpaths)

    def create_knn_model(self, imgdir, pca, bowdict):
        labels, feats,imgpaths = self.create_bow_feature(imgdir, pca,bowdict)
        X = np.zeros((len(feats), feats[0].shape[1]))
        Y = np.zeros((len(feats),))
        for k in range(len(feats)):
            X[k,:] = feats[k]
            Y[k] = labels[k]
        knn = KNeighborsClassifier().fit(X,Y)
        return knn

    def run_knn_predict(self, imgdir, pca, bowdict,knn):
        labels, feats, imgpaths = self.create_bow_feature(imgdir, pca,bowdict)
        X = np.zeros((len(feats), feats[0].shape[1]))
        for k in range(len(feats)):
            X[k,:] = feats[k]
        Y = knn.predict(X)
        hr = 0
        for k in range(len(labels)):
            if Y[k] == labels[k]:
                hr += 1
        hr = hr * 1.0 / len(labels)
        return (imgpaths, Y,hr)

    def run(self,rootdir):
        traindir = rootdir+"train/"
        testdir = rootdir+'test/'
        samplepath = rootdir+"samples.txt"
        pcapath = rootdir+'pca.txt'
        dictpath = rootdir+"dict.txt"

        #create pca,dict models
        self.extract_raw_feature(traindir,samplepath,pcapath)
        self.create_bow_dict(samplepath, pcapath,dictpath)

        with open(pcapath, 'r') as f:
            pca = pickle.load(f)

        with open(dictpath, 'r') as f:
            bowdict = pickle.load(f)

        #create knn models
        knn = self.create_knn_model(traindir, pca, bowdict)

        #do testing
        imgpaths, Y, hr = self.run_knn_predict(testdir, pca, bowdict, knn)
        print 'hr = ',hr


if __name__ == "__main__":
    with open('config.txt', 'r') as f:
        for line in f:
            rootdir = line.strip()
            break
    spider = IMAGE_SPIDER()        
    spider.run(rootdir)



