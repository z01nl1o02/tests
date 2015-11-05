import os,sys,pdb,cv2
import numpy as np

class FEAT_HOG(object):
    def __init__(self):
        self.hog = cv2.HOGDescriptor((64,64),(16,16),(8,8),(8,8),9)

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
        std_sz = (64,64)
        img = cv2.resize(img,std_sz,interpolation=cv2.cv.CV_INTER_LINEAR)
        feat = self.hog.compute(img)
        fv = np.reshape(feat, (1,-1)).tolist()[0]
        return fv

    def folder_mode(self,paths, count):
        fvs = []
        bspaths = self.bootstrap(paths, count)
        for fname in bspaths:
            fv = self.image_mode(fname)
            fvs.append(fv) 
        return (fvs, bspaths)

    def bootstrap(self, paths,count):
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
    fv = FEAT_HOG().image_mode('a.jpg')
    print len(fv)
   
   
   
   
   
   
   
   
   
   
   
   
    
