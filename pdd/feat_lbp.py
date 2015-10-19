import os,sys,pdb,cv2
import numpy as np
from skimage.feature import local_binary_pattern

class FEAT_LBP(object):
    def image_mode(self,imagepath):
        img = cv2.imread(imagepath,0)
        if img is None:
            print imagepath + " is null"
            return (None, None)
        std_sz = (60,30)
        img = cv2.resize(img,std_sz)
        lbp = local_binary_pattern(img, 8, 1, 'uniform')
        fv = []
        bins = np.arange(60)
        for y in range(0, lbp.shape[0], 10):
            for x in range(0, lbp.shape[1], 10):
                blk = lbp[y:y+10, x:x+10]
                h = np.histogram(blk, bins)[0]/100.0
                fv.extend(h)
        return fv

    def folder_mode(self,rootpath):
        fvs = []
        paths = []
        for rdir, pdir, names in os.walk(rootpath):
            for name in names:
                sname,ext = os.path.splitext(name)
                if 0 != cmp(ext, '.jpg'):
                    continue
                fname = os.path.join(rdir, name)
                fv = self.image_mode(fname)
                paths.append(name)
                fvs.append(fv) 
        return (fvs, paths)


