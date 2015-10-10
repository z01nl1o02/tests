import os,sys,pdb,pickle,cv2
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import image_gabor_feature as igbf


def KMeans_A(rootdir, posdir,posnum,negnum_p):
    pos = []
    neg = [] 
    pathpos = []
    pathneg = []
    folders = []
    imgspos = []
    imgsneg = []
    folders = [posdir] #only check the pointed folder
    gbf = igbf.GABOR_FEAT()
    for folder in folders:
        fname = os.path.join(rootdir, folder)
        if 0 == cmp(folder, posdir):
            fvs,imgs = gbf.gen_folder(fname, posnum)
            if fvs is None:
                print 'pos None ',fname
                continue
            pos.extend(fvs)
            imgspos.extend(imgs)
            pathpos.extend([folder for k in range(len(fvs))])
        else:
            fvs,imgs = gbf.gen_folder(fname, negnum_p)
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
    com_num = np.minimum(100, samples.shape[0] - 10)
    clf = PCA(com_num)
    print 'before pca : ', samples.shape
    samples = clf.fit_transform(samples)
    print 'after pca : ', samples.shape
    clf = KMeans(n_clusters=4,n_jobs=-2)
    prds = clf.fit_predict(samples)
    line0 = ""
    line1 = ""
    line2 = ""
    line3 = ""
    for k in range(len(prds)):
        if prds[k] == 0:
            line0 += imgs[k] + '\n'
        elif prds[k] == 1:
            line1 += imgs[k] + '\n'
        elif prds[k] == 2:
            line2 += imgs[k] + '\n'
        else:
            line3 += imgs[k] + '\n'
    with open('A.txt', 'w') as f:
        f.writelines(line0)
    with open('B.txt', 'w') as f:
        f.writelines(line1)
    with open('C.txt', 'w') as f:
        f.writelines(line2)
    with open('D.txt', 'w') as f:
        f.writelines(line3)
    return 

if __name__=="__main__":
    with open('config.txt','r') as f:
        rootdir = f.readline().strip()
        posdir = f.readline().strip()
        posnum = np.int64(f.readline().strip())
        negnum_p = np.int64(f.readline().strip())
    posnum = 5000
    negnum_p = 5000
    KMeans_A(rootdir, posdir,posnum, negnum_p)







