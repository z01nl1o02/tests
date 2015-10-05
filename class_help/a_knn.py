import os,sys,pdb,pickle,cv2
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
import image_gabor_feature as igbf


def KNN_A(rootdir, posdir, posnum, negnum_p):
    pos = []
    neg = [] 
    pathpos = []
    pathneg = []
    folders = []
    imgspos = []
    imgsneg = []
    with open('list.txt', 'r') as f:
        for line in f:
            line = line.strip()
            folders.append(line)
    gbf = igbf.GABOR_FEAT()
    for folder in folders:
        fname = os.path.join(rootdir, folder)
        if 0 == cmp(folder, posdir):
            fvs,imgs = gbf.gen_folder_gabor(fname, posnum)
            if fvs is None:
                print 'pos None ',fname
                continue
            pos.extend(fvs)
            imgspos.extend(imgs)
            pathpos.extend([folder for k in range(len(fvs))])
        else:
            fvs,imgs = gbf.gen_folder_gabor(fname, negnum_p)
            if fvs is None:
                print 'neg None ', fname
                continue
            neg.extend(fvs)
            imgsneg.extend(imgs)
            pathneg.extend([folder for k in range(len(fvs))])
    label0 = [0 for k in range(len(pos))]
    label1 = [1 for k in range(len(neg))]
    samples = np.array(pos + neg)
    labels = np.array(label0 + label1)
    paths = pathpos + pathneg
    imgs = imgspos + imgsneg
    clf = PCA(100)
    print 'before pca : ', samples.shape
    samples = clf.fit_transform(samples)
    print 'after pca : ', samples.shape
    if 0:
        clf = KNeighborsClassifier(5)
        clf.fit(samples,labels)

        res = [] 
        for k in range(samples.shape[0]):
            prd = clf.predict(samples[k,:])
            res.append((paths[k],prd))
        res = sorted(res, key = lambda k : k[0])
        line = ""
        for path, prd in res:
            line += path + ' ' + str(prd) + '\n'
        with open('result.txt', 'w') as f:
            f.writelines(line)
    else:
        clf = NearestNeighbors(5).fit(samples)
        dists,idxs = clf.kneighbors(samples, 5)
        line = ""
        for k in range(len(idxs)):
            for j in range(len(idxs[k])):
                line += paths[idxs[k][j]] + ' '
            line += '\n'
        with open('result.txt', 'w') as f:
            f.writelines(line)
    return 

if __name__=="__main__":
    with open('config.txt','r') as f:
        rootdir = f.readline().strip()
        posdir = f.readline().strip()
        posnum = np.int64(f.readline().strip())
        negnum_p = np.int64(f.readline().strip())
    KNN_A(rootdir, posdir, posnum, negnum_p)







