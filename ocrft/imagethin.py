#reference: http://www.cnblogs.com/xianglan/archive/2011/01/01/1923779.html


import cv2
import copy
import pdb
import numpy as np

class IMAGETHIN:
    def __init__(self):
        self._array = \
        [0,0,1,1,0,0,1,1,1,1,0,1,1,1,0,1,\
         1,1,0,0,1,1,1,1,0,0,0,0,0,0,0,1,\
         0,0,1,1,0,0,1,1,1,1,0,1,1,1,0,1,\
         1,1,0,0,1,1,1,1,0,0,0,0,0,0,0,1,\
         1,1,0,0,1,1,0,0,0,0,0,0,0,0,0,0,\
         0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,\
         1,1,0,0,1,1,0,0,1,1,0,1,1,1,0,1,\
         0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,\
         0,0,1,1,0,0,1,1,1,1,0,1,1,1,0,1,\
         1,1,0,0,1,1,1,1,0,0,0,0,0,0,0,1,\
         0,0,1,1,0,0,1,1,1,1,0,1,1,1,0,1,\
         1,1,0,0,1,1,1,1,0,0,0,0,0,0,0,0,\
         1,1,0,0,1,1,0,0,0,0,0,0,0,0,0,0,\
         1,1,0,0,1,1,1,1,0,0,0,0,0,0,0,0,\
         1,1,0,0,1,1,0,0,1,1,0,1,1,1,0,0,\
         1,1,0,0,1,1,1,0,1,1,0,0,1,0,0,0]
             
    def thinV(self,image,array):
        h,w = image.shape
        NEXT = 1
        for i in range(h):
            for j in range(w):
                if NEXT == 0:
                    NEXT = 1
                else:
                    if j > 0 and j < w - 1:
                        M = np.int64(image[i,j-1]) + np.int64(image[i,j]) + np.int64(image[i,j+1])
                    else:
                        M = 1
                    if image[i,j] == 0  and M != 0:                  
                        a = [0]*9
                        for k in range(3):
                            for l in range(3):
                                if -1<(i-1+k)<h and -1<(j-1+l)<w and image[i-1+k,j-1+l]==255:
                                    a[k*3+l] = 1
                        sum = a[0]*1+a[1]*2+a[2]*4+a[3]*8+a[5]*16+a[6]*32+a[7]*64+a[8]*128
                        image[i,j] = array[sum]*255
                        if array[sum] == 1:
                            NEXT = 0
        return image
        
    def thinH(self,image,array):
        h,w = image.shape
        NEXT = 1
        for j in range(w):
            for i in range(h):
                if NEXT == 0:
                    NEXT = 1
                else:
                    M = np.int64(image[i-1,j]) + np.int64(image[i,j]) + np.int64(image[i+1,j]) if 0<i<h-1 else 1   
                    if image[i,j] == 0 and M != 0:                  
                        a = [0]*9
                        for k in range(3):
                            for l in range(3):
                                if -1<(i-1+k)<h and -1<(j-1+l)<w and image[i-1+k,j-1+l]==255:
                                    a[k*3+l] = 1
                        sum = a[0]*1+a[1]*2+a[2]*4+a[3]*8+a[5]*16+a[6]*32+a[7]*64+a[8]*128
                        image[i,j] = array[sum]*255
                        if array[sum] == 1:
                            NEXT = 0
        return image
        
    def thin(self,BW,num=100):
        img = copy.deepcopy(BW)
        for i in range(num):
            self.thinV(img,self._array)
            self.thinH(img,self._array)
        return img


if __name__=="__main__":   
    img = cv2.imread('1.bmp',0)
    th,BW = cv2.threshold(img,0,255,cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    BW = 255 - BW #dark text on light backgound
    thin = IMAGETHIN().thin(BW)
    print img.shape, BW.shape, thin.shape
    #pdb.set_trace()
    cv2.imshow('img',img)
    cv2.imshow('BW',BW)
    cv2.imshow('thin',thin)
    cv2.waitKey(-1)
