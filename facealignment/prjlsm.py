import os,sys,pdb,pickle
import cv2
import numpy as np
import argparse
import re

class PRJLSM(object):
    def __init__(self):
        self.rootdir='lfw'
        self.annfile = 'LFW_annotation.txt'
        self.faces = self.load_annotation()
        self.canthusrr = 0
        self.canthusrr = 1
        self.canthuslr = 2
        self.canthusll = 3
        self.mouthr = 4
        self.mouthl = 5
        self.nose = 6
        self.ptsnum = 7
        self.meanxy = self.calc_mean_shape()

    def warp_im(self,im, M, dshape):
        output_im = np.zeros(dshape, dtype=im.dtype)
        cv2.warpAffine(im, M[:2], (dshape[1], dshape[0]), dst = output_im, borderMode=cv2.BORDER_TRANSPARENT,
        flags = cv2.WARP_INVERSE_MAP)
        return output_im



    def transformation(self,points_dst, points_src):
        #transform from points_src to points_dst
        points_dst = np.matrix(points_dst)
        points_src = np.matrix(points_src)
        points_dst = points_dst.astype(np.float64)
        points_src = points_src.astype(np.float64)
        
        c1 = np.mean(points_dst, axis=0)
        c2 = np.mean(points_src, axis=0)
        points_dst -= c1
        points_src -= c2

        s1 = np.std(points_dst)
        s2 = np.std(points_src)
        points_dst /= s1
        points_src /= s2

        #Orthogonal Procrustes Problem
        U,S,Vt = np.linalg.svd(points_dst.T * points_src)
        R = (U * Vt).T
        return np.vstack( [np.hstack(((s2 / s1) * R, c2.T - (s2/ s1) * R * c1.T)), np.matrix([0.,0.,1.]) ] )




    def load_annotation(self):
        faces = {}
        with open(self.annfile, 'rb') as f:
            for line in f:
                data = re.split(r'\s+',line)[0:-1]
                img = data[0]
                rect = [np.int32(d) for d in data[1:5]]
                total = len(data)
                pts_x = [np.float32(data[k]) for k in range(5,total,2)]
                pts_y = [np.float32(data[k]) for k in range(6,total,2)]
                if img not in faces:
                    person = img[0:-9]
                    #convert to relative xy
                    w = rect[2] - rect[0]
                    h = rect[3] - rect[1]
                    for k in range(len(pts_x)):
                        pts_x[k] = (pts_x[k] - rect[0]) / w
                    for k in range(len(pts_y)):
                        pts_y[k] = (pts_y[k] - rect[1]) / h
                    faces[img] = [person, rect, pts_x,pts_y]
                else:
                    print 'dupliate image %s'%img
        return faces
    def calc_mean_shape(self):
        total = len(self.faces)
        ptsnum = self.ptsnum
        X = np.zeros((total, ptsnum))
        Y = np.zeros((total, ptsnum))
        k = 0
        for img in self.faces:
            pts_x = self.faces[img][2]
            pts_y = self.faces[img][3]
            X[k,:] = np.array(pts_x)
            Y[k,:] = np.array(pts_y)
            k += 1
        meanx = np.mean(X, 0).tolist()
        meany = np.mean(Y, 0).tolist()
        return (meanx, meany)

    def draw_landmark(self, img, pts_x, pts_y):
        for k in range(len(pts_x)):
            x = np.int32(pts_x[k])
            y = np.int32(pts_y[k])
            cv2.circle(img, (x,y), 3, color=(0,255,0))
        return img
    def test(self):
        imgs = [ 'Al_Gore_0004.jpg', 'Al_Gore_0006.jpg' ]
        ptslist = []
        for img in imgs:
            person = self.faces[img][0]
            rect = self.faces[img][1]
            w = rect[2] - rect[0]
            h = rect[3] - rect[1]
            X = self.faces[img][2] 
            Y = self.faces[img][3]
            pts = np.zeros((self.ptsnum, 2))
            for k in range(len(X)):
                x = X[k]
                y = Y[k]
                pts[k,0] = x * w + rect[0]
                pts[k,1] = y * h + rect[1] #restore to origin coordinates
            srcpath = os.path.join(self.rootdir,person)
            srcpath = os.path.join(srcpath, img)
            im = cv2.imread(srcpath)
            im = self.draw_landmark(im, pts[:,0], pts[:,1])
            ptslist.append((im,pts))
        M = self.transformation(ptslist[0][1], ptslist[1][1])
        output_im = self.warp_im(ptslist[1][0], M, ptslist[1][0].shape)

        for k in range(2):
            cv2.imshow(imgs[k], ptslist[k][0])
        cv2.imshow('output', output_im)
        cv2.waitKey(-1) 
if __name__=="__main__":
    prjlsm = PRJLSM()
    prjlsm.test()


