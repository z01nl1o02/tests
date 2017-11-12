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
        return (x**2) * math.log(x)
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
    def run(self,pts_from, pts_to):
        self.control_pts = copy.deepcopy(pts_from)
        self.Xs = []
        #calc K (it's shared computation)
        total = len(pts_from)
        K = np.zeros( (total,total))
        for i in range(total):
            for j in range(i,total):
                a = np.asarray(pts_from[i])
                b = np.asarray(pts_to[j])
                K[i,j] = self.radian_func(np.linalg.norm(a - b))
                K[j,i] = K[i,j]
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
    def calc_surface(self,W,H):
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
            m1 = F.max()
            m0 = F.min()
            print m0,m1
            scale = 1.0
            if m1 > m0:
                scale = 1.0 / (m1 - m0)
            canvas = np.zeros((H,W))
            for y in range(H):
                for x in range(W):
                    val = (F[y * W + x] - m0) * 255 * scale
                    canvas[y,x] = val
            canvas = np.uint8(canvas)
            cv2.imshow("surface",canvas)
            cv2.waitKey(-1)
        return


if __name__=="__main__":
    tps = TPS2D()
    pts_from = []
    pts_to = []
    for k in range(4):
        x = k * 10
        y = k * 4
        pts_from.append((x,y))
        pts_to.append((x + 10, y))
    tps.run(pts_from, pts_to)
    tps.calc_surface(255,255)
    



