import os,sys,pdb,cv2
import numpy as np

class FEAT_YUV(object):
    def image_mode(self,imagepath):
        img = cv2.imread(imagepath,1)
        if img is None:
            print imagepath + " is null"
            return (None, None)
        std_sz = (60,20)
        img = cv2.resize(img,std_sz)
        fv = [0 for k in range(32*32*3)]
        for y in range(img.shape[0]):
            for x in range(img.shape[1]):
                Y = img[y,x,0] / 8
                Cr = img[y,x,1] / 8
                Cb = img[y,x,2] / 8
                fv[Y * 32 + Cb] += 1
                fv[Y * 32 + Cr+1024] += 1
                fv[Cr * 32 + Cb + 2048] += 1
        sz = img.shape[0] * img.shape[1]
        for k in range(len(fv)):
            fv[k] = fv[k] * 1.0 / sz
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
       
   
   
   
   
   
   
   
   
   
   
   
   
    
