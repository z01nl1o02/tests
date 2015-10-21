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
        std_sz = (128,64)
        img = cv2.resize(img,std_sz)
        feat = self.hog.computer(img)
        fv = np.reshape(feat,(1,-1)).tolist()[0]
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


