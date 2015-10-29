import os,sys,pdb,pickle,cv2
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import image_gabor_feature as igbf
import image_lbp_feature as ilbpf
import image_hog_feature as ihogf
import image_dwt_feature as  idwtf

def KMeans_A(rootdir,ft):
    pos = []
    imgspos = []
    if 0 == cmp(ft, 'lbp'):
        print "ft : LBP"
        gbf=ilbpf.LBP_FEAT()
    elif 0 == cmp(ft, 'gabor'):
        print "ft : GABOR"
        gbf = igbf.GABOR_FEAT()
    elif 0 == cmp(ft, 'hog'):
        print 'ft : HOG'
        gbf = ihogf.HOG_FEAT()
    elif 0 == cmp(ft, 'dwt'):
        print 'ft : DWT'
        gbf = idwtf.DWT_FEAT()
    else:
        print 'unknown ft'
        return 
    fvs,imgs = gbf.gen_folder(rootdir, 5000)
    if fvs is None:
        print 'JPG None ',rootdir
        return
    pos.extend(fvs)
    imgspos.extend(imgs)
    samples = np.array(pos)
    imgs = imgspos
    com_num = np.minimum(300, samples.shape[0] - 10)
    clf = PCA(com_num)
    print 'before pca : ', samples.shape
    samples = clf.fit_transform(samples)
    print 'after pca : ', samples.shape
    clf = KMeans(n_clusters=2,n_jobs=-2,verbose = 0)
    prds = clf.fit_predict(samples)
    line0 = ""
    line1 = ""
   # line2 = ""
   # line3 = ""
    for k in range(len(prds)):
        if prds[k] == 0:
            line0 += imgs[k] + '\n'
        elif prds[k] == 1:
            line1 += imgs[k] + '\n'
   #     elif prds[k] == 2:
   #         line2 += imgs[k] + '\n'
   #     else:
   #         line3 += imgs[k] + '\n'
    with open('A.txt', 'w') as f:
        f.writelines(line0)
    with open('B.txt', 'w') as f:
        f.writelines(line1)
   # with open('C.txt', 'w') as f:
   #     f.writelines(line2)
   # with open('D.txt', 'w') as f:
   #     f.writelines(line3)
    return 

if __name__=="__main__":
    if len(sys.argv) == 3:
        ft = sys.argv[1]
        rootdir = sys.argv[2]
        KMeans_A(rootdir,ft)
    else:
        print 'unknown option'







