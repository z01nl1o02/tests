import os,sys,pdb
import cv2
import numpy as np
import math

img = np.zeros((128,128),dtype=np.uint8)

cx = 64
cy = 64
Ah = 40
Bh = 20
R = 30 * math.pi / 180.0

sin_r = math.sin(R)
cos_r = math.cos(R)
for t in range(0,3600,1):
    t = t * math.pi / (10.0 * 180.0)
    x0 = Ah * math.cos(t)
    y0 = Bh * math.sin(t)
    x = cos_r * x0 + sin_r * y0
    y = -sin_r * x0 + cos_r * y0
    x = np.int64(round(x)) + cx
    y = np.int64(round(y)) + cy
    img[y,x] = 1

for y in range(img.shape[0]):
    x0 = 0
    for x in range(img.shape[1]):
        if img[y,x] == 1:
            x0 = x
            break
    x1 = img.shape[1] - 1
    for x in range(img.shape[1]-1,0,-1):
        if img[y,x] == 1:
            x1 = x
            break
    if x0 == 0 and x1 == img.shape[1] - 1:
        continue
    img[y,x0:x1] = 1

cv2.imwrite('img.jpg',img*255)
sumx = 0
sumy = 0
sumxy = 0
sumxx = 0
sumyy = 0
num = 0
for y in range(img.shape[0]):
    for x in range(img.shape[1]):
        if img[y,x] == 0:
            continue
        sumx += x
        sumy += y
        sumxy += (x - cx) * (y - cy)
        sumxx += (x - cx) * (x - cx)
        sumyy += (y - cy) * (y - cy)
        num += 1
mx = sumx / num
my = sumy / num
mxy = -sumxy/num #right-hand coordinates
mxx = sumxx/num
myy = sumyy/num

delta = math.sqrt(mxy * mxy * 4 + (mxx - myy)**2)
#eigen value of matrix
# mxx   mxy
# mxy   myy
a = math.sqrt((mxx + myy + delta) / 2.0)
b = math.sqrt((mxx + myy - delta) / 2.0)
print a*2,b*2
print Ah,Bh


Rd = math.atan2(2*mxy,mxx - myy)
print Rd * 180 / (2*math.pi)
print R*180/ math.pi


cv2.imshow('img',img * 255)
cv2.waitKey(-1)



