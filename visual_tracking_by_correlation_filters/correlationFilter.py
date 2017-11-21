import os,sys,pdb
import numpy as np
import cv2
import math
import matplotlib.pyplot as plt
import copy


def gen_G(W, H, center, sigma):
    x0 = center[0] * 1.0
    y0 = center[1] * 1.0
    g = np.zeros((H,W))
    for y in range(H):
        for x in range(W):
            g[y,x] = math.exp( -1 * ( (x-x0)**2 + (y-y0)**2 ) / (2 * sigma**2))
    g /= g.sum()
    return g

def cvtFreq(F):
    F = np.log(np.abs(F) + 1)
    m0 = F.min()
    m1 = F.max()
    F = 255 * (F - m0) / (m1 - m0)
    return np.uint8(F)

def preprocess(I):
    I = (I - I.min()) / (I.max() - I.min())
    return I


def show_gray_with_mark(winname, gray, pt,r, rgb=(255,0,0)):
    color = cv2.cvtColor(gray,cv2.COLOR_GRAY2BGR)
    cv2.circle(color, pt, r, rgb)
    cv2.imshow(winname,color)

def show_train_sample(winname,f,g,F,G, pt, r, rgb=(255,0,0)):
    f_img = 255 * (f - f.min()) / (f.max() - f.min())
    canvasF = np.hstack( (np.uint8(f_img), cvtFreq(F)) ) 
    canvasG = np.hstack( (np.uint8(g/g.max() * 255), cvtFreq(G)))
    canvas = np.vstack((canvasF, canvasG))
    canvas = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)
    cv2.circle(canvas,pt, r,rgb)
    cv2.imshow(winname,canvas)
    return

#how to train one correlation filter from initial frame
def corrFilter(testimage, targetPt, targetR):
    img = cv2.imread(testimage,0)
    show_gray_with_mark("origin", img, targetPt, targetR)
    cx,cy = targetPt
    R = targetR
    FF = []
    GF = []
    for dx in range(-5,5,5):
        for dy in range(-5,5,5):
            for theta in range(-10,10,10):
                for scale in range(8,12,1):
                    #affine
                    affMat = cv2.getRotationMatrix2D((cx + dx,cy+dy),theta,scale/10.0)
                    affRes = cv2.warpAffine(img,affMat,(img.shape[1],img.shape[0]))
                    f = affRes * 1.0 
                    #input
                    f = preprocess(f)

                    #target
                    g = gen_G(img.shape[1],img.shape[0],(cx+dx,cy+dy),2.0)
                    F = ( np.fft.fft2(f) )
                    G = ( np.fft.fft2(g) )
                    FF.append(F * F.conjugate())
                    GF.append(G * F.conjugate())

                    #show current input and output
                    show_train_sample("sample",f,g,F,G,(cx + dx, cy+dy), 5, (0,255,0))
                    cv2.waitKey(100)
    H = reduce(lambda a,b:a + b, GF) / reduce(lambda a,b: a + b, FF)
    H = np.real(H)
   
    #convert H from frequence to space
    h = np.abs( np.fft.ifft2(H) )
    h = np.uint8( 255 * (h - h.min()) / (h.max() - h.min()))
    canvas = np.hstack((h,cvtFreq(H)))
    cv2.imshow('correlation-filter(h-H)',canvas)
    cv2.waitKey()
    return H


def test(testimage, H):
    f = cv2.imread(testimage,0) * 1.0
    f = preprocess(f)
    F = np.fft.fft2(f) 
    G = F * H.conjugate()

    g = np.abs( np.fft.ifft2(G) )
    v0,v1,l0,l1 = cv2.minMaxLoc(g)
    x,y = l1 #get maximum as detection result
    show_gray_with_mark('test-det',cv2.imread(testimage,0), (x,y), 8, (0,255,255))
    g = (g - g.min()) * 255 / (g.max() - g.min())
    canvas = np.hstack((cvtFreq(G),np.uint8(g)))
    cv2.imshow("test-g-G", canvas)
    cv2.waitKey()



def run():
    #pt = (126,164)
    pt = (96,127)
    R = 32
    H = corrFilter("2.jpg", pt,R)
    test("2.jpg",H)

if __name__=="__main__":
    run()


