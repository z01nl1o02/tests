import os,sys,pdb
import numpy as np
import cv2
import math
class IMWARP(object):
    def __init__(self,scale,rotation,stretch,dist,pan):
        self.w,self.h = [np.float64(x) for x in scale]
        self.rotX, self.rotY, self.rotZ = [np.float64(x) for x in rotation]
        self.stretchX, self.stretchY = [np.float64(x) for x in stretch]
        self.dist = np.float64(dist)
        self.panX,self.panY = [np.float64(x) for x in pan]
        return
    def get_transform(self,scale,rotation,stretch,dist,pan):
        w,h = scale
        rotX,rotY,rotZ = rotation
        stretchX,stretchY = stretch
        panX,panY = pan
        F = 1.0
        w = np.float64(w)
        h = np.float64(h)
        #projection 2D->3D 
        A1 = [ 1, 0, -w/2,
                      0, 1, -h/2,
                      0, 0,     0,
                      0, 0,     1]
        A1 = np.asarray(A1)
        A1.shape = (4,3)
        #camera Intrisecs matrix 3D->2D
        A2 = [ F, 0, w/2, 0,
                   0, F, h/2, 0,
                   0, 0, 1,  0]
        A2 = np.asarray(A2)
        A2.shape = (3,4)
        #rotation around x-axis
        sin = math.sin(rotX)
        cos = math.cos(rotX)
        Rx = [
                   1,      0,           0,       0,
                   0,    cos,       -sin,    0,
                   0,    sin,       cos,     0,
                   0,    0,           0,         1]
        Rx = np.asarray(Rx)
        Rx.shape = (4,4)
        #rotation aound y-axis
        sin = math.sin(rotY)
        cos = math.cos(rotY)
        Ry = [
                   cos,  0,  sin,  0,
                   0,     1,    0,    0,
                   -sin, 0,  cos, 0,
                   0,     0,    0,    1]
        Ry = np.asarray(Ry)
        Ry.shape = (4,4)
        #rotation around z-axis
        sin = math.sin(rotZ)
        cos = math.cos(rotZ)
        Rz = [ 
                   cos,  -sin, 0, 0,
                   sin,   cos,  0, 0,
                   0,        0,    1,  0,
                   0,        0,    0,  1]
        Rz = np.asarray(Rz)
        Rz.shape = (4,4)
        R = Rx.dot( Ry.dot(Rz) )
        T = [
                 stretchX, 0,   0,    panX,
                 0             , 1,   0,   panY,
                 0            ,  0,   1,  dist,
                 0            , 0,    0,  1]
        T = np.asarray(T)
        T.shape = (4,4)
        return A2.dot( T.dot( R.dot(A1) ) )
    def warp_image(self,img):
        H,W,C = img.shape
        widthRatio = self.w / np.float64(W)
        heightRatio = self.h / np.float64(H)
        rotX = self.rotX * widthRatio
        rotY = self.rotY * heightRatio
        panX = self.panX / widthRatio
        panY = self.panY  / heightRatio

        transform = self.get_transform( (W,H),(rotX,rotY,self.rotZ),
                (self.stretchX,self.stretchY),
                self.dist,
                (panX,panY))
        result = cv2.warpPerspective(img, transform, dsize=(W,H),
                flags=cv2.INTER_CUBIC | cv2.WARP_INVERSE_MAP)
        return result

img = None
imwarp = None

def update_image():
    result = imwarp.warp_image(img)
    cv2.imshow('result',result)
    return 
def upate_rotx(value):
    global imwarp
    imwarp.rotX = (value - 50) / 2000.0
    update_image()
    return

def upate_roty(value):
    global imwarp
    imwarp.rotY = (value-50) / 2000.0
    update_image()
    return

def upate_rotz(value):
    global imwarp
    imwarp.rotZ = (value-50) / 50.0 * math.pi
    update_image()
    return


def upate_dist(value):
    global imwarp
    imwarp.dist = 1 + (value - 50) / 25.0
    update_image()
    return

def upate_stretchx(value):
    global imwarp
    imwarp.stretchX =  1 + (value - 50)/10.0
    update_image()
    return

def run(imgpath):
    global img
    img = cv2.imread(imgpath,1)
    H,W,C = img.shape
    global imwarp
    imwarp = IMWARP((W,H), (0,0,0), (1.0,1.0),1,(0,0))
    cv2.imshow('src', img)
    cv2.createTrackbar('rotX','src',50,100,upate_rotx)
    cv2.createTrackbar('rotY','src',50,100,upate_roty)
    cv2.createTrackbar('rotZ','src',50,100,upate_rotz)
    cv2.createTrackbar('dist','src',50,100,upate_dist)
    cv2.createTrackbar('stretchX','src',50,100,upate_stretchx)
    update_image()
    cv2.waitKey(-1)
    return

if __name__=="__main__":
    run(sys.argv[1])



