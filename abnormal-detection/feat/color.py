import os,sys,pdb
import numpy
import cv2

def get(img):
    img = cv2.resize(img,(64,64)) * 1.0 / 255
    blue =img[:,:,0] 
    green = img[:,:,1]
    red = img[:,:,2]
    bm = cv2.moments(blue, False)
    gm = cv2.moments(green,False)
    rm = cv2.moments(red,False)
    feat = []
    for y in range(4):
        for x in range(4):
            if x + y >= 3:
                continue
            name = 'nu%d%d'%(x,y)
            if name not in rm.keys():
                continue
            feat.append( rm['nu%d%d'%(x,y)] )
            feat.append( gm['nu%d%d'%(x,y)] )
            feat.append( bm['nu%d%d'%(x,y)] )
    return feat

if __name__=="__main__":
    img = cv2.imread('c:/tmp/plate.jpg',1)
    feat = get(img)
    print len(feat)
    print feat


