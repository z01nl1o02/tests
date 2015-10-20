import os,sys,pdb,pickle,cv2,math
import numpy as np
import pywt

class DWT_FEAT(object):
    def __init__(self):
        self.resample_w = 128
        self.resample_h = 64

    def crop_image(self, imgpath): 
        name = imgpath.split('\\')[-1]
        sname,ext = os.path.splitext(name)
        brand,x0,y0,x1,y1,plate = sname.split(',')
        x0 = np.int64(x0)
        x1 = np.int64(x1)
        y0 = np.int64(y0)
        y1 = np.int64(y1)
        if x1 == 0 and y1 == 0:
            return None
        w = x1 - x0 + 1
        h = y1 - y0 + 1
        left = x0 - w - w/2
        right = x1 + w + w/2
        top = y0 - w
        bottom = y1 + h
        w = right - left + 1
        h = bottom - top + 1
        subimg = np.zeros((h,w))
        img = cv2.imread(imgpath,0)
        if left < 0 or top < 0 or right >= img.shape[1] or bottom >= img.shape[0]:
            return None
        subimg = img[top:bottom, left:right]
        subimg = cv2.resize(subimg,(self.resample_w,self.resample_h))
        return subimg

    def gen_image(self, imgpath):
        img = self.crop_image(imgpath)
        if img is None:
            return None
        feats = []
        coefs = pywt.wavedec2(img, 'db2', level=1)
        cA3 = coefs[0]
        cH3 = coefs[1][0]
        cV3 = coefs[1][1]
        cD3 = coefs[1][2]
        feats.append(cA3)
        feats.append(cH3)
        feats.append(cV3)
        feats.append(cD3)

        fv = []
        for feat in feats:
            feat = np.reshape(feat,(1,-1)).tolist()[0]
            fv.extend(feat)
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
    imgpath = '1,379,1118,496,1155,1.jpg'
    fv = DWT_FEAT().gen_image(imgpath)
        
    





