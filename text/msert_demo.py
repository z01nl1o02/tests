import os,sys,pdb,pickle,cv2
import numpy as np
def test_mser(imgpath,outpath):
    img = cv2.imread(imgpath, 1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mser = cv2.MSER()
    regs = mser.detect(gray, None)
    hulls = [cv2.convexHull(p.reshape(-1,1,2)) for p in regs]
    cv2.polylines(img, hulls, 1, (0,255,0))
    cv2.imwrite(outpath, img)

    os.mkdir('int')
    num = 0
    for reg in regs:
        mask = np.zeros(gray.shape)
        for idx in range(reg.shape[0]):
            x = reg[idx,0]
            y = reg[idx,1]
            mask[y,x] = 255
        mask = np.uint8(mask)
        cv2.imwrite('int/'+str(num)+'.jpg', mask)
        num += 1

if __name__=="__main__":
    try:
        test_mser(sys.argv[1],'result.jpg')
    except Exception, e:
        print e




