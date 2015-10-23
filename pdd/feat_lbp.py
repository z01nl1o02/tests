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
        xnum = 3
        ynum = 3
        blkw = img.shape[1] / xnum
        blkh = img.shape[0] / ynum
        bins = np.arange(59)
        for y in range(0, lbp.shape[0], blkh):
            for x in range(0, lbp.shape[1], blkw):
                blk = lbp[y:y+blkh, x:x+blkw]
                h = np.histogram(blk, bins)[0] * 1.0 / (blkw * blkh)
                fv.extend(h)
        return fv

    def folder_mode(self,rootpath, count):
        fvs = []
        paths = []
        bspaths = self.bootstrap(rootpath, count)
        for name, fname in bspaths:
            fv = self.image_mode(fname)
            paths.append(name)
            fvs.append(fv) 
        return (fvs, paths)
    def bootstrap(self, rootpath, count):
        paths = []
        for rdir,pdir,names in os.walk(rootpath):
            for name in names:
                sname,ext = os.path.splitext(name)
                if 0 != cmp(ext, '.jpg'):
                    continue
                fname = os.path.join(rdir, name)
                paths.append((name,fname))
        if count < 0:
            bspaths = paths
        else:
            idxs = np.random.randint(len(paths), size = count)
            bspaths = []
            for k  in range(count):
                idx = idxs[k]
                bspaths.append(paths[idx])
        return bspaths
       
   
   
   
   
   
   
   
   
   
   
   
   
    
