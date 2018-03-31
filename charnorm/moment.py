import os,sys,pdb
import numpy as np
import cv2
import math

def calc_params(img):
    H,W  = img.shape
    m00 = img.sum()
    m10 = 0
    m01 = 0
    for y in range(H):
        for x in range(W):
            m10 += x * img[y,x]
            m01 += y * img[y,x]
    cx = m10 / m00
    cy = m01 / m00
    u11 = 0
    u20 = 0
    u02 = 0

    for y in range(H):
        for x in range(W):
            u11 += (x-cx) * (y - cy) * img[y,x]
            u20 += (x-cx) * (x-cx) * img[y,x]
            u02 += (y-cy) * (y - cy) * img[y,x]
    u11 /= float(m00)
    u20 /= float(m00)
    u02 /= float(m00)
    delta = math.sqrt( u11 * u11 * 4 + (u20 - u02)**2 )
    a = math.sqrt ( (u20 + u02 + delta) / 2.0)
    b = math.sqrt( (u20 + u02 - delta) / 2.0)
    R = math.atan2(2*u11, u20-u02)
    return (cx,cy, a, b, R/2)

def norm(img, params):
    cx,cy,a,b,R = params
    print 'rotated: ', R * 180 / math.pi
    if R < 0: #make rotated alone one-direction
        R = math.pi + R

    stdH,stdW = (128*2,128*2)
    H,W = img.shape
    new = np.zeros((stdH,stdW))
    cos = math.cos(R)
    sin = math.sin(R)
    for y in range(stdH):
        for x in range(stdW):
            sy0 = (y - stdH/2.0) / stdH * H
            sx0 = (x - stdW/2.0)/ stdW * W
            sx = cos * sx0 - sin * sy0
            sy = sin * sx0 + cos * sy0
            sx += cx
            sy += cy
            sy = np.int64(sy)
            sx = np.int64(sx)
            if sx < 0 or sy < 0 or sx >= W or sy >= H:
                continue
            new[y,x] = img[sy,sx]
    cv2.imshow('norm',np.uint8(new))
    cv2.imshow('src',img)
    cv2.waitKey(-1)

img = cv2.imread('img.jpg',0)
params = calc_params(img)
norm(img, params)



