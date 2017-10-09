import os,sys,pdb
import numpy as np
import cv2
import math
import imagethin
import copy

class SHAPE_FEAT:
    def __init__(self,debugFlag = False):
        self._debugFlag = debugFlag
        self._blockW = 12
        self._blockH = 12
        self._eohSize = 6
    def raw_contour(self,filepath):
        img = cv2.imread(filepath,0)
        th,bw = cv2.threshold(img,0,255,cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
        #bw = cv2.Canny(bw,10,100)
        bw = 255 - bw
        bw = imagethin.IMAGETHIN().thin(bw)
        if self._debugFlag:
            cv2.imwrite("bw.jpg",255-bw)
        return 255-bw
    def simulate_contour(self,bw_):
       # bw = cv2.GaussianBlur(bw_,(3,3),2)
        bw = copy.deepcopy(bw_)
        dx = cv2.Sobel(bw,cv2.CV_32F,1,0)
        dy = cv2.Sobel(bw,cv2.CV_32F,0,1)
        if self._debugFlag:
            cv2.imwrite('dx.jpg',dx)
            cv2.imwrite('dy.jpg',dy)
        H,W = bw.shape
        res = []
        for y in range(0,H,self._blockH):
            for x in range(0,W,self._blockW):
                m0 = 0
                m1x = 0
                m1y = 0
                for row in range(y,y+self._blockH):
                    for col in range(x,x+self._blockW):
                        if row >= H or col >= W:
                            continue
                        if bw[row,col] == 0:
                            continue
                        m0 += 1
                        m1x += col
                        m1y += row
                if m0 > 0:
                    cx = m1x / m0
                    cy = m1y / m0
                else:
                    continue
                relax = 1
                eohsize = self._eohSize
                oristep = 180.0 / eohsize
                eoh = np.asarray([0.0 for k in range(eohsize)])
                for row in range(y,y+self._blockH):
                    for col in range(x,x+self._blockW):
                        if row >= H or col >= W:
                            continue
                        #if col == 51 and row == 14:
                        #    pdb.set_trace()
                        if relax > 0 and row - relax > 0 and row + relax < H and col - relax > 0 and col + relax < W:
                            lx = 1.0*dx[row-relax:row+relax,col-relax:col+relax].mean()
                            ly = 1.0*dy[row-relax:row+relax,col-relax:col+relax].mean()
                        else:
                            lx = 1.0*dx[row,col]
                            ly = 1.0*dy[row,col] #rotate (dx,dy) by 90 degree to get gradient
                        if 1:                         
                            gx = ly
                            gy = lx
                        mag = math.sqrt( gx ** 2 + gy ** 2)
                        if mag < 0.001:
                            continue
                        theta = math.atan2(gy,gx) * 180 / math.pi
                        if theta < 0:
                            theta += 180
                        theta = np.int64( theta / oristep)
                        #print gx,gy,theta
                        if theta >= eohsize:
                            theta = 0
                        eoh[theta] += mag
                if eoh.sum() > 0:
                    eoh = eoh / eoh.sum()
                    #ori = (eoh * np.asarray([oristep*k * math.pi/180.0 for k in range(eohsize)])).sum()
                    #pdb.set_trace()
                    ori = np.argmax(eoh) * math.pi / eohsize
                    res.append([cx,cy,ori])
        return res

    def draw_shape(self,shapes,img):
        canvas = np.zeros(img.shape,np.uint8)
        H,W = canvas.shape
        for shape in shapes:
            cx,cy,ori = shape
            if cx < 0 or cy < 0:
                continue
            cos = math.cos(ori)
            sin = math.sin(ori)
            cx = np.int64(cx)
            cy = np.int64(cy)
            y0 = cy
            for x0 in range(cx - self._blockW/2, cx + self._blockW/2):
                x1 = x0 - cx
                y1 = y0 - cy
                x2 = x1 * cos + y1 * sin
                y2 = -x1 * sin + y1 * cos
                x2 += cx
                y2 += cy
                x2 = np.int64(x2)
                y2 = np.int64(y2)
                if x2 < 0 or y2 < 0 or x2 >= W or y2 >= H:
                    continue
                canvas[y2,x2] = 255
        if self._debugFlag:
            cv2.imwrite('canvas.jpg',canvas)
        return canvas



if __name__=="__main__":
    sf = SHAPE_FEAT(True)
    bw = sf.raw_contour('1.bmp')
    res = sf.simulate_contour(bw)
    canvas = sf.draw_shape(res,bw)
    cv2.waitKey(-1)



