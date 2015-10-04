import os,sys,pdb,pickle,cv2,math
import numpy as np
import gabor2d


class GABOR_FEAT(object):
    def __init__(self):
        self.resample_w = 90
        self.resample_h = 60
        self.gabors = []
        for w in range(5,21,5):
            for a in range(0,180,30):
                angle = a * math.pi / 180.0
                gb = gabor2d.create_gabor_2d(1,1,0,w,angle)
                self.gabors.append(gb)

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
        left = x0 - w
        right = x1 + w
        top = y0 - 2 * w
        bottom = y0 
        w = right - left + 1
        h = bottom - top + 1
        subimg = np.zeros((h,w))
        img = cv2.imread(imgpath,0)
        if left < 0 or top < 0 or right >= img.shape[1] or bottom >= img.shape[0]:
            return None
        subimg = img[top:bottom, left:right]
        subimg = cv2.resize(subimg,(self.resample_w,self.resample_h))
        return subimg

    def gen_image_gabor(self, imgpath):
        img = self.crop_image(imgpath)
        if img is None:
            return None
        feats = []
        for gb in self.gabors:
            feat = cv2.filter2D(img,cv2.CV_64F,gb) 
            feats.append(feat)
        fv = []
        for feat in feats:
            for y in range(0, feat.shape[0], 10):
                for x in range(0, feat.shape[1],10):
                    fv.append(feat[y:y+10,x:x+10].mean())
        return fv

    def gen_folder_gabor(self, rootdir, capacity):
        fvs = []
        imgs = []
        for rdir,pdir,names in os.walk(rootdir):
            for name in names:
                sname,ext = os.path.splitext(name)
                if 0 != cmp('.jpg', ext):
                    continue
                fname = os.path.join(rdir, name)
                fv = self.gen_image_gabor(fname)
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
    fv = GABOR_FEAT().gen_image_gabor(imgpath)
        
    





