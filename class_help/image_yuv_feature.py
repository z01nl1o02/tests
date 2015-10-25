import os,sys,pdb,pickle,cv2,math
import numpy as np
from skimage.feature import local_binary_pattern

#it is a special one wrt other feature in croping

class YUV_FEAT(object):
    def __init__(self):
        self.resample_w = 60
        self.resample_h = 30

    def crop_image(self, imgpath): 
        #not crop and color requred
        img = cv2.imread(imgpath,1)
        img = cv2.cvtColor(img, cv2.cv.CV_BGR2YCrCb)
        img = cv2.resize(img,(self.resample_w,self.resample_h))
        return img

    def gen_image(self, imgpath):
        img = self.crop_image(imgpath)
        if img is None:
            return None

        fv = [0 for k in range(64 * 32)]
        for y in range(img.shape[0]):
            for x in range(img.shape[1]):
                Y = img[y,x,0] / 4
                Cr = img[y,x,1] / 8
                Cb = img[y,x,2] / 8
                c = Y * 32 + Cr
                fv[c] += 1
        normfactor = 1.0 / (img.shape[0] * img.shape[1])
        for k in range(len(fv)):
            fv[k] = fv[k] * normfactor
        return fv

    def gen_folder(self, rootdir, capacity):
        rootdir = rootdir.strip('\\')
        fvs = []
        imgs = []
        for rdir,pdir,names in os.walk(rootdir):
            if  len(rdir.strip('\\')) != len(rootdir):
                print 'jump images in ', rdir
                continue
            for name in names:
                sname,ext = os.path.splitext(name)
                if 0 != cmp('.jpg', ext):
                    continue
                fname = os.path.join(rdir, name)
                fv = self.gen_image(fname)
                if fv is None:
                    continue
                fvs.append(fv)
                imgs.append(sname)
                if len(fvs) >= capacity:
                    return (fvs,imgs)
        if len(fvs) == 0:
            return (None,None)
        else:
            return (fvs,imgs)     

if __name__=="__main__":
    #imgpath = '1,379,1118,496,1155,1.jpg'
    imgpath = 'a.jpg'
    fv = YUV_FEAT().gen_image(imgpath)
        
    





