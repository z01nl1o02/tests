#https://en.wikipedia.org/wiki/Procrustes_analysis
import os,sys,pdb,cPickle
import cv2
import numpy as np
import copy
import math

with open('shape.pkl','rb') as f:
    landmarks = cPickle.load(f)

data = np.zeros( (len(landmarks), len(landmarks[0]) ) )

for k,pts in enumerate(landmarks):
    data[k,:] = np.asarray(pts)

ptsNum = len(landmarks[0]) / 2
shapeNum = len(landmarks)

src = copy.deepcopy(data)
for trynum in range(10):
    #translation
    srcT = np.zeros( src.shape)
    for k in range(shapeNum):
        mx = np.mean( src[k,0:ptsNum] )
        my = np.mean( src[k,ptsNum:] )
        srcT[k,0:ptsNum] = src[k,0:ptsNum] - mx
        srcT[k,ptsNum:] = src[k,ptsNum:] - my
    #scaling
    srcTS = np.zeros(src.shape)
    for k in range(shapeNum):
        stdx = np.std( srcT[k,0:ptsNum] )
        stdy = np.std( srcT[k,ptsNum:] )
        srcTS[k,0:ptsNum] = srcT[k,0:ptsNum] / stdx
        srcTS[k,ptsNum:] = srcT[k,ptsNum:] / stdy
    #rotation
    srcTSR = np.zeros(src.shape)
    refShape = srcTS[0,:]
    refX = refShape[0:ptsNum]
    refY = refShape[ptsNum:]
    for k in range(0,shapeNum):
        X = srcTS[k,0:ptsNum]
        Y = srcTS[k,ptsNum:]
        A = (X * refY - Y * refX).sum()
        B = (X * refX + Y * refY).sum()
        theta = math.atan2(A,B)
        cos = math.cos(theta)
        sin = math.sin(theta)
        srcTSR[k,0:ptsNum] = X * cos - Y * sin
        srcTSR[k,ptsNum:] = X * sin + Y * cos
    changed = np.abs((srcTSR - src)).sum()
    print 'iter',trynum,'changed',changed
    src = copy.deepcopy(srcTSR)
    if changed < 0.001:
        break
canvasGray = np.zeros( (500,500), dtype = np.uint8)
scale = canvasGray.shape[0] / 5
cx = canvasGray.shape[1] / 2
cy = canvasGray.shape[0] / 2
for k in range(src.shape[0]):
    canvas = cv2.cvtColor(canvasGray,cv2.COLOR_GRAY2BGR)
    shape = src[k,:]
    X = shape[0:ptsNum] * scale + cx 
    Y = shape[ptsNum:] * scale + cy
    for x, y in zip(X,Y):
        x = np.int64(x)
        y = np.int64(y)
        cv2.circle(canvas,(x,y),2,(0,255,255))
    cv2.imshow('result',canvas)
    cv2.waitKey(100)





















