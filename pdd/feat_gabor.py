import os,sys,pdb,cv2,math
import numpy as np
import gabor2d

class FEAT_GABOR(object):
    def __init__(self):
        self.stdw = 64
        self.stdh = 64
        self.gabors = []
        for w in range(5,21,5):
            for a in range(0,180,30):
                angle = a * math.pi / 180.0
                gb = gabor2d.create_gabor_2d(1,1,0,w,angle)
                self.gabors.append(gb)

    def image_mode(self,imagepath):
        img = cv2.imread(imagepath,0)
        if img is None:
            print imagepath + " is null"
            return (None, None)
        x0 = np.int32(img.shape[1]/4)
        x1 = np.int32(img.shape[1]*3/4)
        y0 = np.int32(img.shape[0]/10)
        y1 = np.int32(img.shape[0] * 2 / 3)
        img = cv2.blur(img,(3,3),0.5)
        img = img[y0:y1, x0:x1] #focus it
        std_sz = (self.stdh,self.stdw)
        img = cv2.resize(img,std_sz,interpolation=cv2.cv.CV_INTER_LINEAR)
        feats = []
        for gb in self.gabors:
            feat = cv2.filter2D(img, cv2.CV_64F, gb)
            feats.append(feat)
        fv = []
        for feat in feats:
            for y in range(0, feat.shape[0], 8):
                for x in range(0, feat.shape[1], 8):
                    fv.append( feat[y:y+8, x:x+8].mean() )
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

if __name__=="__main__":
    fv = FEAT_GABOR().image_mode('a.jpg')
    print len(fv)
   
   
   
   
   
   
   
   
   
   
   
   
    
