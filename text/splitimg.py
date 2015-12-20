import os,sys,pdb,pickle,cv2
import numpy as np
def split_rectK(imgpath,minw, maxw, stepw, steph, h2w):
    img = cv2.imread(imgpath,1)
    subimgs = []
    for w in range(minw, maxw, stepw):
        h = np.int64(w * h2w)
        if h < 10:
            continue
        for y in range(0, img.shape[0] - h, steph):
            for x in range(0, img.shape[1] - w, stepw):
                subimg = img[y:y+h, x:x+w, :]
                subimgs.append(subimg)
    return subimgs

def split_rect(indir, outdir, minw = 100, maxw = 180, stepw = 20, steph = 20, h2w = 0.33):
    for root, dirs, names in os.walk(indir):
        for name in names:
            sname,ext = os.path.splitext(name)
            ext.lower()
            if 0 != cmp(ext, '.jpg') and 0 != cmp(ext, '.jpeg'):
                continue 
            print sname
            imgs = split_rectK(os.path.join(root, name), minw, maxw, stepw, steph, h2w)
            for k in range(len(imgs)):
                img = imgs[k]
                outpath = os.path.join(outdir, sname+str(k)+'.jpg')
                cv2.imwrite(outpath, img)
    return


if __name__=='__main__':
    if len(sys.argv) == 4 and 0 == cmp(sys.argv[1], '-rect'):
        indir = sys.argv[2]
        outdir = sys.argv[3]
        split_rect(indir, outdir)

