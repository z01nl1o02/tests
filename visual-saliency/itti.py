import os,sys,pdb
import numpy as np
import cv2
import gabor2d
import math
from collections import defaultdict

def get_image(arr):
    m0 = arr.min()
    m1 = arr.max()
    return np.uint8( 255 * ( arr - m0) / (m1 - m0) )

img = cv2.imread('test.jpg',1)
img = cv2.cvtColor(img,cv2.COLOR_BGR2YUV)
H,W,C = img.shape
pyd = defaultdict(lambda: defaultdict(list)   )
sigmas = []
for ch in range(3):
    tmp = np.reshape( img[:,:,ch], (H,W))
    pyd[ch][0].append( np.float32(tmp) )
    for level in range(8):
        src  = np.uint8(pyd[ch][level][0])
        src =  cv2.GaussianBlur(src, (3,3),1.0)
        h0,w0 = src.shape
        h1 = np.int64(h0/2)
        w1 = np.int64(w0/2)
        if w1 < 24 or h1 < 24:
            break
        dst = cv2.resize(src, (w1,h1))
        dst = np.float32(dst)
        pyd[ch][level + 1].append(dst)

print 'pyd level = %d'%len(pyd[0])
if 0:
    for ch in pyd.keys():
        for level in pyd[ch].keys():
            for tmp in pyd[ch][level]:
                tmp = np.uint8( tmp )
                cv2.imshow('pyd', tmp)
                cv2.waitKey(-1)



gabors = []
for a in range(0,180,45):
    wl = 9
    gabor = gabor2d.create_gabor_2d(1,1,0,wl,math.pi * a / 180.0)
    gabors.append(gabor)
#    cv2.imshow("gabor",  get_image(gabor))
#    cv2.waitKey(-1)

for ch in pyd.keys():
    for level in pyd[ch].keys():
        tmp = np.uint8(pyd[ch][level][0])
        for ori,gabor in enumerate(gabors):
            res = cv2.filter2D(tmp, cv2.CV_32F, gabor)
            res = np.float32(res )
            res = 255 * (res - res.min()) / (res.max() - res.min())
            pyd[ch][level].append(res)

if 0:
    for ch in pyd.keys():
        for level in pyd[ch].keys():
            for tmp in pyd[ch][level]:
                tmp = np.uint8( tmp )
                cv2.imshow('pyd', tmp)
                cv2.waitKey(-1)


levelStep = 2
spyd = []
M = 200
salmaps = defaultdict( lambda: defaultdict(list) )
for ch in pyd.keys():
    for level in pyd[ch].keys():
        if level + levelStep >= len(pyd[ch]):
            continue
        for ori in range( len(pyd[ch][level]) ):
            A = pyd[ch][level][ori]
            B = pyd[ch][level+levelStep][ori]
            A = cv2.resize(A,(W,H))
            B = cv2.resize(B, (W,H))
            C = np.absolute(A - B)
            C = M * (C - C.min())/(C.max() - C.min())

            D = np.uint8(C)
            E = cv2.dilate(D, np.ones((5,5)))
            #cv2.imshow('D',D)
            #cv2.imshow('E',E)
            #cv2.waitKey(-1)

            y,x= np.where(D >= E)
            F = filter(lambda X: X < M and X > M/2,D[y,x].tolist())
            m = np.asarray(F).mean()
            #print np.asarray(F).min(), np.asarray(F).max()
            D = np.float32(D) * (M - m)**2

            salmaps[ch][level].append( D )
            spyd.append( D)
#        tmp = get_image(D)
#        cv2.imshow('spyd',tmp)
#        cv2.waitKey(-1)


saliency_grad = []
saliency_gray = []
saliency_color = []

for ch in salmaps.keys():
    for level in salmaps[ch].keys():
        for ori in range(1, len(salmaps[ch][level])):
            saliency_grad.append( salmaps[ch][level][ori] )

for level in salmaps[0].keys():
    saliency_gray.append( salmaps[0][level][0] )

for ch in salmaps.keys():
    if ch == 0:
        continue
    for level in salmaps[ch].keys():
        saliency_color.append( salmaps[ch][level][0] )


saliency_grad = reduce( lambda X,Y: X+Y, saliency_grad) / len(saliency_grad)
saliency_gray = reduce( lambda X,Y: X+Y, saliency_gray) / len(saliency_gray)
saliency_color = reduce( lambda X,Y: X+Y, saliency_color) / len(saliency_color)

saliency_map = (saliency_grad + saliency_gray + saliency_color) / 3
saliency_map = get_image(saliency_map)
cv2.imshow('saliency',saliency_map)
cv2.imshow('img',cv2.cvtColor(img,cv2.COLOR_YUV2BGR))
cv2.waitKey(-1)








