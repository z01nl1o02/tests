import os,sys,pdb,pickle,cv2
import mlpbase as mb
import numpy as np
import gabor2d
import math

class FEAT(object):
    def __init__(self):
        self.stdw = 32
        self.stdh = 64
        self.flts = []
        for w in range(5, 21, 5):
            for a in range(0, 180, 30):
                angle = a * math.pi / 180.0
                gb = gabor2d.create_gabor_2d(1,1,0,w,angle)
                self.flts.append(gb)
    def feat_for_image(self, imgpath):
        feat = None
        try:
            img = cv2.imread(imgpath, 0)
            img = cv2.resize(img,(self.stdh, self.stdw))
            layers = []
            for flt in self.flts:
                layer = cv2.filter2D(img, cv2.CV_64F, flt)
                layers.append(layer)
            stepx = self.stdw / 4
            stepy = self.stdh / 8
            feat = []
            for layer in layers:
                for y in range(0, layer.shape[0], stepy):
                    for x in range(0, layer.shape[1], stepx):
                        feat.append( layer[y:y+stepy, x:x+stepx].mean())
        except Exception, e:
            feat = None
            print 'exception :', e ,':',imgpath
        return feat      
    
def load_samples(dirpath, classid):
    feats = []
    featmaker = FEAT()
    for root, dirs, names in os.walk(dirpath):
        for name in names:
            sname,ext = os.path.splitext(name)
            if 0 != cmp(ext, '.jpg'):
                continue
            fname = os.path.join(root, name)
            feat = featmaker.feat_for_image(fname)
            if None == feat:
                continue
            feats.append(feat)

    total = len(feats)
    dim = len(feats[0])
    target_list = [[classid] for k in range(total)]
    return feats, target_list

def load_all_sample(dirpath):
    feats = []
    target_list = []
    for k in range(4):
        fs, tgtlist = load_samples(os.path.join(dirpath, str(k)), k)
        feats.extend(fs)
        target_list.extend(tgtlist)
    return feats, target_list


def demo(rootdir):
    outsize = 4
    mlp = mb.MLP_PROXY('ocrmlp.dat')
    feats, target_list = load_all_sample(os.path.join(rootdir, 'train'))
    samples = np.array(feats)
    targets = mlp.target_vec2mat(target_list, outsize)
    samples, targets = mlp.shuffle(samples, targets)
    insize = samples.shape[1]
    mlp.create([insize, 64, outsize])
    print 'train ', samples.shape, ',', targets.shape
    mlp.train(samples, targets)

    print 'predict...'
    feats, target_list0 = load_all_sample(os.path.join(rootdir, 'test'))
    samples = np.array(feats)
    targets = mlp.predict(samples)
    target_list1 = mlp.target_mat2vec(targets,outsize)
    hit = 0
    for a,b in zip(target_list0, target_list1):
        if len(a) == len(b) and len(a) == 1 and a[0] == b[0]:
            hit += 1
    print len(target_list1), ',', hit, ',', hit * 1.0 / len(target_list) 

if __name__=='__main__':
    demo(sys.argv[1])






    

