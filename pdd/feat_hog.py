import os,sys,pdb,cv2
import numpy as np

class FEAT_HOG(object):
    def __init__(self):
        self.hog = cv2.HOGDescriptor()

    def image_mode(self,imagepath):
        img = cv2.imread(imagepath,0)
        if img is None:
            print imagepath + " is null"
            return (None, None)
        std_sz = (64,128) #can't set winsize for hog so size here is unchangable!!!
        img = cv2.resize(img,std_sz)
        feat = self.hog.compute(img)
        fv = np.reshape(feat, (1,-1)).tolist()[0]
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
       
   
   
   
   
   
   
   
   
   
   
   
   
    
