import os,sys,pdb
import numpy as np
import cv2
import gabor2d
import math
from collections import defaultdict
import copy
from scipy.sparse import csc_matrix
def get_image(arr):
    m0 = arr.min()
    m1 = arr.max()
    return np.uint8( 255 * ( arr - m0) / (m1 - m0) )


img = cv2.imread('test.jpg',1)
H,W,C = img.shape

R = np.reshape( img[:,:,0],(H,W))
G = np.reshape( img[:,:,1],(H,W))
B = np.reshape( img[:,:,2],(H,W))
img = (R+G+B)/3
H,W = img.shape
scale = 1
while(H/scale > 20 or W / scale > 20):
    scale += 1

chn =[]
chn.append(np.float32(R))
chn.append(np.float32(G))
chn.append(np.float32(B))
chn.append(np.float32(img))

pyd = []
pyd.extend(chn)
for s in range(1):
    for ch in chn:
        tmp = math.pow(2,s+1)
        h0,w0 = ch.shape
        h1 = np.int64(h0/tmp)
        w1 = np.int64(w0/tmp)
        pyd.append( np.float32( cv2.resize( np.uint8(ch), (w1,h1)   )) )

if 0:
    wl = 5
    for a in range(0,180,45):
        gabor = gabor2d.create_gabor_2d(1,1,0,wl,math.pi * a / 180.0)
        res = cv2.filter2D( np.uint8(pyd[-1]),  cv2.CV_32F,gabor)
        #res = np.absolute(res)
        res = 255 * (res - res.min()) / (res.max() - res.min())
        pyd.append( np.float32(res) )

w = np.int64(W / scale)
h = np.int64(H / scale)
for k in range(len(pyd)):
    tmp = pyd[k]
    #cv2.imshow('grad', get_image(tmp))
    #cv2.waitKey(-1)
    tmp = cv2.resize(tmp, (w,h))
    pyd[k] = tmp


delta = 5.0
radius = 10
filt = defaultdict(np.int64)
for y in range(radius):
    for x in range(radius):
        dist = y * y + x * x
        filt[dist] = math.exp( -1 * dist / (2*delta**2) )

def get_mf(img, radius, filt):
    H,W = img.shape
    mf = np.zeros( (H*W,H*W), dtype=np.float32   )
    tmp = np.zeros( (H*W,H*W) )
    for y in range(H):
        for x in range(W):
            a0 = img[y,x]
            p0 =  y * W + x
            for dy in range(-radius, radius):
                if y + dy < 0 or y + dy >= H:
                    continue
                for dx in range(-radius, radius):
                    if x + dx < 0 or x  + dx >= W:
                        continue
                    dist = dx ** 2 + dy ** 2
                    w = filt[dist]
                    a1 = img[y+dy,x+dx]
                    p1 = (y+dy) * W + (x+dx)
                    tmp[p0,p1] = w
                    mf[p0,p1] += w * np.absolute(a1-a0)
    for y in range(W * H):
        mf[y,:] /= mf[y,:].sum()
    #tmp = get_image(tmp)
    #cv2.imshow('mf',tmp)
    #cv2.waitKey(-1)
    return mf

def solve_for_equilibrium(mf):
    tol = 0.01
    mfT = mf.transpose()
    H,W = mfT.shape

    prev = np.ones( (H,1),dtype=np.float32) / H
    res = None
    for k in range(100):
        curr = mfT.dot(prev)
        update = np.absolute( prev - curr).sum()
        if update < tol:
            res = prev
            break
        prev = curr / curr.sum()
    print k
    return res,k

def cvt2img(res,shape,W,H):
    res = res / res.max()
    res.shape  = shape
    pic = np.uint8(res * 255)
    pic = cv2.resize(pic,(W, H))
    return pic

respyd = []
for pic in pyd:
    mf = get_mf(pic, radius, filt) 
    res,iterNum = solve_for_equilibrium(mf)
    if res is None:
        print 'failed to equilibrium'
        continue
    respyd.append(res)
    #tmp = cvt2img(res,pic.shape,W,H)
    #cv2.imshow('act', tmp)
    #cv2.imshow('src', cvt2img(pic,pic.shape,W,H))
    #cv2.waitKey(-1)

tmp = reduce( lambda X,Y:X+Y, respyd ) / len(respyd)
tmp = cvt2img(tmp, pyd[0].shape, W,H)
cv2.imshow('sal', tmp)
cv2.waitKey(-1)
