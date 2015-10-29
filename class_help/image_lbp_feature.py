import os,sys,pdb,pickle,cv2,math
import numpy as np
from skimage.feature import local_binary_pattern


class LBP_FEAT(object):
    def __init__(self):
        self.resample_w = 90
        self.resample_h = 60

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
#        img = cv2.GaussianBlur(img,(5,5),1)
        feats = []
        feat = local_binary_pattern(img, 8, 1, 'uniform')
        feats.append(feat)
        feat = local_binary_pattern(img, 8, 3, 'uniform')
        feats.append(feat)
#        feat = local_binary_pattern(img, 8, 5, 'uniform')
#        feats.append(feat)
        fv = []
        bins = [k for k in range(59)]
        for feat in feats:
            xstep = feat.shape[1] / 3
            ystep = feat.shape[0] / 3
            for y in range(0, feat.shape[0], ystep):
                for x in range(0, feat.shape[1],xstep):
                    block = feat[y:y+ystep,x:x+xstep]
                    h = np.histogram(block, bins)[0] * 1.0 / (xstep * ystep)
                    fv.extend(h)
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
    fv = LBP_FEAT().gen_image(imgpath)
        
    





