import os,sys,cv2,pdb
from sklearn.decomposition import TruncatedSVD
from pylab import *

def get_feature(imgpath):
    img = cv2.imread(imgpath,0)
    img = cv2.resize(img,(32,64))
    img = np.float32(img)
    img = img / 255
    img = np.reshape(img, (1,32*64))
    return img

def extract_sample_from_image(imgdir):
    feats = []
    for rdir, pdir, names in os.walk(imgdir+'pos'):
        for name in names:
            sname,ext = os.path.splitext(name)
            if 0 == cmp(ext, '.jpg'):
                fname = os.path.join(rdir, name)
                feats.append(get_feature(fname))
    for rdir, pdir, names in os.walk(imgdir+'neg'):
        for name in names:
            sname,ext = os.path.splitext(name)
            if 0 == cmp(ext, '.jpg'):
                fname = os.path.join(rdir, name)
                feats.append(get_feature(fname))
    sample_num = len(feats)
    sample_size = feats[0].shape[1]
    samples = np.zeros((sample_num, sample_size))
    for k in range(sample_num):
        samples[k,:] = feats[k]
    print 'samples ', samples.shape[0], samples.shape[1]
    return samples
     
def run_svd(samples):
    svd = TruncatedSVD(2)
    svd.fit(samples)
    newsamples = svd.transform(samples)
    return (svd, newsamples)

def show_svd(transformed):
    sample_num = transformed.shape[0]
    for k in range(sample_num):
        if k*2<sample_num:
            mark = 'rx'
        else:
            mark = 'bo'
        x,y = (transformed[k,0], transformed[k,1])
        plot(x,y,mark)
    show()

if __name__=="__main__":
    samples = extract_sample_from_image('img/') 
    svd, transformed = run_svd(samples)
    show_svd(transformed)

   
