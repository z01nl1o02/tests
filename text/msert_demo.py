import os,sys,pdb,pickle,cv2
import numpy as np
import shutil
class MSER_DEMO(object):
    def mser_gray(self,gray):
        mser = cv2.MSER()
        regions = mser.detect(gray,None)
        return regions
    
    def mser_color(self,bgr):
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV) 
        rgns = []
        rgns.extend( self.mser_gray(hsv[:,:,1]) )
        rgns.extend( self.mser_gray(255 - hsv[:,:,1]) )
        rgns.extend( self.mser_gray(hsv[:,:,2]) )
        rgns.extend( self.mser_gray(255 - hsv[:,:,2]) )
        return rgns
    
    def show_mser(self,bgr):
        rgns = self.mser_color(bgr)
        hulls = [cv2.convexHull(p.reshape(-1,1,2)) for p in rgns]
        cv2.polylines(bgr, hulls, 1, (0,255,0))
        cv2.imwrite('result.jpg', bgr)

        try:
            shutil.rmtree('result/')
        except Exception, e:
            print 'rmtree exception ', e

        try: 
            os.mkdir('result')
        except Exception, e:
            print 'mkdir exception ', e
        num = 0
        for reg in rgns:
            mask = np.zeros(bgr.shape)
            for idx in range(reg.shape[0]):
                x = reg[idx,0]
                y = reg[idx,1]
                mask[y,x,0] = 255
                mask[y,x,1] = 255
                mask[y,x,2] = 255
            mask = np.uint8(mask)
            cv2.imwrite('result/'+str(num)+'.jpg', mask)
            num += 1
         
def test_mser(imgpath,outpath):
    img = cv2.imread(imgpath, 1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mser = cv2.MSER()
    regs = mser.detect(gray, None)
    pdb.set_trace()
    hulls = [cv2.convexHull(p.reshape(-1,1,2)) for p in regs]
    cv2.polylines(img, hulls, 1, (0,255,0))
    cv2.imwrite(outpath, img)

    os.mkdir('int')
  
if __name__=="__main__":
    try:
        img = cv2.imread(sys.argv[1])
        MSER_DEMO().show_mser(img)
    except Exception, e:
        print e




