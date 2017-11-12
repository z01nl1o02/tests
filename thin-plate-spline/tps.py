"""
https://elonen.iki.fi/code/tpsdemo/index.html
http://profs.etsmtl.ca/hlombaert/thinplates/#eq:thinplates
"""
import os,sys,pdb
import numpy as np
import cv2
import copy
import math
class TPS2D(object):
    def __init__(self):
        self.Xs = []
        self.control_pts = []
    def radian_func(self,x):
        if x <= 0:
            return 0
        return (x**2) * math.log(x) / math.log(10)
    def run_one_dimension(self, pts3D,K):
        #calc P
        total = len(pts3D)
        P = np.zeros( (total,3) )
        for k in range(total):
            P[k,0] = 1
            P[k,1] = pts3D[k][0]
            P[k,2] = pts3D[k][1]
        #calc Y
        Y = np.zeros( (total + 3,1) )
        for k in range(total):
            Y[k,0] = pts3D[k][2]
        #calc L
        L0 = np.hstack((K,P))
        L1 = np.hstack((np.transpose(P),np.zeros((3,3))))
        L = np.vstack((L0,L1))
        #solve linear system LX = Y
        X = np.linalg.solve(L,Y)
        return X
    def run(self,pts_from, pts_to,regulation=1.0):
        self.control_pts = copy.deepcopy(pts_from)
        self.Xs = []
        #calc K (it's shared computation)
        total = len(pts_from)
        K = np.zeros( (total,total))
        alpha = 0.0
        for i in range(total):
            for j in range(i + 1,total):
                a = np.asarray(pts_from[i])
                b = np.asarray(pts_from[j])
                val = np.linalg.norm(a-b)
                K[i,j] = self.radian_func(val)
                K[j,i] = K[i,j]
                alpha += val * 2 
        alpha /= (total ** 2)
        for i in range(total):
            K[i,i] = alpha * alpha * regulation
        #x-dimension
        pts3D = []
        for k in range(total):
            a = pts_from[k]
            b = pts_to[k]
            pts3D.append( [a[0],a[1],b[0] - a[0]] )
        self.Xs.append( self.run_one_dimension(pts3D,K) )
        #y-dimension
        pts3D = []
        for k in range(total):
            a = pts_from[k]
            b = pts_to[k]
            pts3D.append( [a[0],a[1],b[1] - a[1]] )
        self.Xs.append( self.run_one_dimension(pts3D,K) )
    def calc_surface(self,gray):
        H,W = gray.shape
        offsets = []
        for X in self.Xs:
            data = np.zeros((W*H,3+len(self.control_pts)))
            for y in range(H):
                for x in range(W):
                    p1 = np.asarray([x,y])
                    for k, pts in enumerate(self.control_pts):
                        p0 = np.asarray(pts)
                        data[y * W + x,k] = self.radian_func( np.linalg.norm( p0- p1 ) )
                    data[y * W + x, k + 1] = 1
                    data[y * W + x, k + 2] = x
                    data[y * W + x, k + 3] = y
            F = np.dot(data, X )
            offset = np.zeros((H,W))
            for y in range(H):
                for x in range(W):
                    offset[y,x] = F[y * W + x]
            offsets.append(offset)
        canvas = np.zeros((H,W))
        for y in range(H):
            for x in range(W):
                y0 = np.int64(y + offsets[1][y,x])
                x0 = np.int64(x + offsets[0][y,x])
                if x0 < 0 or y0 < 0 or x0 >= W or y0 >= H:
                    continue
                canvas[y,x] = gray[y0,x0]
        cv2.imshow('result',np.uint8(canvas))
        color = cv2.cvtColor(gray,cv2.COLOR_GRAY2BGR)
        for pts in self.control_pts:
            cv2.circle(color,pts,8,(0,0,255))
        cv2.imshow('source',color)
        cv2.waitKey(-1)
        return


if __name__=="__main__":
    tps = TPS2D()
    gray = cv2.imread('timg.jpg',0)
    H,W = gray.shape
    pts_from = []
    pts_to = []
    pts_from.append( (10,10) )
    pts_from.append( (W-10,H-10))
    pts_from.append( (W-10,10))
    pts_from.append( (10,H-10))
    pts_to.extend(pts_from)

    pts_from.append( (W/2, H/2))
    pts_to.append( (W/2 + 100, H/2-50))

    pts_from.append( (W/4, H/4))
    pts_to.append( (W//4 - 10, H/4-10))

    tps.run(pts_from, pts_to,0)
    tps.calc_surface(gray)
    



