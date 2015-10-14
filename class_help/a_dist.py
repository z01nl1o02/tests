import os,sys,pdb,pickle,cv2,shutil
import numpy as np
from sklearn.decomposition import PCA
import image_gabor_feature as igbf
import image_lbp_feature as ilbpf
import mklist
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
        score = sorted(score, key = lambda k: k[0],reverse = False)
        scores.append(score[0:topN])
    return scores

#fromdir_root = d:\
#fromdir = a
#todir_root = c:
#todir = 1
#copy/rename image d:\a\a1.jpg to c:\1\[a]a1.jpg
def copy_rename_jpgs(fromdir_root, fromdir, todir_root,todir, total):
    num = 0
    srcdir = os.path.join(fromdir_root, fromdir)
    for rdir,pdirs, names in os.walk(srcdir):
        if len(rdir.strip('\\').split('\\')) != len(srcdir.strip('\\').split('\\')):
            print 'jump sub-folder ', rdir
            continue
        if num >= total:
            break
        for name in names:
            sname,ext = os.path.splitext(name)
            if 0 != cmp(ext, '.jpg'):
                continue
            old = os.path.join(rdir, name)
            new = os.path.join(todir_root, todir)
            if 0 == cmp(fromdir, todir):
                new = os.path.join(new,'A['+fromdir+']'+name)
            else:
                new = os.path.join(new,'B['+fromdir+']'+name)
            shutil.copy(old,new)
            num += 1
            if num >= total:
                break
    return

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
    con_num = 300
    if con_num >= len(imgs):
        con_num = len(imgs) - 10
    clf = PCA(con_num)
    samples = clf.fit_transform(samples)
    print 'after pca : ', samples.shape
    dists = calc_distance_set(samples, samples, posnum * 2)

    smap = {}
    for k in range(samples.shape[0]):
        pospath = paths[k]
        if pospath not in smap:
            smap[pospath] = {}
        for dist,j in dists[k]:
            negpath = paths[j]
            if negpath not in smap[pospath]:
                smap[pospath][negpath] = 99999.0
            if smap[pospath][negpath] > dist:
                smap[pospath][negpath] = dist

    slist = sorted(smap.iteritems(), key=lambda k:k[0])
    
    line = ""
    for pospath, negpaths in slist:
        line += pospath + '\n'
        negpaths = sorted(negpaths.iteritems(), key = lambda k:k[1],reverse=False)
        for negpath,cnt in negpaths:
            line += '    ' + negpath + '(' + str(cnt) + ')\n'

    with open('result.txt', 'w') as f:
        f.writelines(line) 

  
    try: 
        shutil.rmtree('out/')
    except Exception as e:
        print 'catch exception ', e

    os.mkdir('out/')
    for pospath,negpaths in slist:
        print 'create viewset for ',pospath
        os.mkdir('out/'+pospath)
        for negpath in negpaths.keys():
            copy_rename_jpgs(rootdir, negpath, 'out\\', pospath, 4)         
    return 



#get all similarity match in folderB compared with folderA
def DIST_B(rootdir, folderA, folderB,ft):
    pos = []
    neg = [] 
    imgspos = []
    imgsneg = []
    if 0 == cmp(ft, 'gabor'):
        print 'feature type: GABOR'
        gbf = igbf.GABOR_FEAT()
    else:
        print 'feature type: LBP'
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
        mklist.gen_dir_list(rootdir)
        posnum = 10
        negnum_p = 10
        DIST_A(rootdir, posdir, posnum, negnum_p,ft)
    elif len(sys.argv) == 5:
        ft = sys.argv[1]
        rootdir = sys.argv[2]
        folderA = sys.argv[3]
        folderB = sys.argv[4]
        DIST_B(rootdir, folderA, folderB,ft)
    else:
        print 'unknown options'





