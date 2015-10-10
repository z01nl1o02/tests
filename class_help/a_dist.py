import os,sys,pdb,pickle,cv2
import numpy as np
from sklearn.decomposition import PCA
import image_gabor_feature as igbf
import image_lbp_feature as ilbpf
#distance based image match

def calc_distance(spl, tmp, distname):
    if 0 == cmp(distname, 'cosine'):
        a = (spl * tmp * 1.0)
        spl = spl ** 2
        tmp = tmp ** 2
        b = np.sqrt(spl.sum()) * np.sqrt(tmp.sum()) + 0.0001
        return 1 - np.abs(a.sum()) / b
    return -9999 

#for each spl in samples, to find the topN matched tmp in templ
def calc_distance_set(samples, templ, topN):
    scores = []
    for k in range(samples.shape[0]):
        spl = samples[k,:]
        score = []
        for j in range(templ.shape[0]):
            tmp = templ[j,:]
            d = calc_distance(spl,tmp, 'cosine')
            score.append( (d, j) )
        score = sorted(score, key = lambda k: k[0])
        scores.append(score[0:topN])
    return scores


def DIST_A(rootdir, posdir, posnum, negnum_p, ft):
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
    if 0 == cmp(ft, 'gabor'):
        print 'feature type: GABOR'
        gbf = igbf.GABOR_FEAT()
    else:
        print 'feature type: LBP'
        gbf = ilbpf.LBP_FEAT()
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
    samples = np.array(pos + neg)
    paths = pathpos + pathneg
    imgs = imgspos + imgsneg
    clf = PCA(300)
    samples = clf.fit_transform(samples)
    print 'after pca : ', samples.shape
    dists = calc_distance_set(samples, samples, posnum * 2)

    line = ""
    for k in range(samples.shape[0]):
        pospath = paths[k]
        line += pospath + '\n'
        for dist,j in dists[k]:
            line += '    ' + paths[j] + '(' + str(dist) + ')\n'

    with open('result.txt', 'w') as f:
        f.writelines(line) 

    return 



#get all similarity match in folderB compared with folderA
def DIST_B(rootdir, folderA, folderB,ft):
    pos = []
    neg = [] 
    imgspos = []
    imgsneg = []
    print 'feature type: ', ft
    if 0 == cmp(ft, 'gabor'):
        gbf = igbf.GABOR_FEAT()
    else:
        gbf = ilbpf.LBP_FEAT()

    #1--folderA
    fvs, imgs = gbf.gen_folder(os.path.join(rootdir, folderA), 1000)
    pos.extend(fvs)
    imgspos.extend(imgs)
    #2--folderB
    fvs, imgs = gbf.gen_folder(os.path.join(rootdir, folderB), 1000)
    neg.extend(fvs)
    imgsneg.extend(imgs)

    #3--match
    samples = np.array(pos + neg)
    imgs = imgspos + imgsneg
    com_num = 300
    if com_num + 10 > len(imgs):
        com_num = len(imgs) - 10
    clf_pca = PCA(com_num)
    samples = clf_pca.fit_transform(samples)
    print 'after pca : ', samples.shape
    dists = calc_distance_set(samples[0:len(pos),:], samples[len(pos):,:],5)

    templist = []
    for dist in dists:
        for d, j in dist:
            templist.append(imgsneg[j])
    templist = list(set(templist))
    lineA = ""
    for imgs in templist:
        lineA += imgs + '\n' 
    with open('A.txt', 'w') as f:
        f.writelines(lineA)
    return 



if __name__=="__main__":
    if len(sys.argv) == 1:
        with open('config.txt','r') as f:
            rootdir = f.readline().strip()
            posdir = f.readline().strip()
            posnum = np.int64(f.readline().strip())
            negnum_p = np.int64(f.readline().strip())
            ft = f.readline().strip()
        posnum = 3
        negnum_p = 3
        DIST_A(rootdir, posdir, posnum, negnum_p,ft)
    elif len(sys.argv) == 5:
        ft = sys.argv[1]
        rootdir = sys.argv[2]
        folderA = sys.argv[3]
        folderB = sys.argv[4]
        DIST_B(rootdir, folderA, folderB,ft)
    else:
        print 'unknown options'





