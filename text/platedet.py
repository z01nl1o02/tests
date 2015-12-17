import os,sys,pdb,pickle,cv2,shutil
import numpy as np
import mlpbase


def gen_feat(imgpath):
    img = cv2.imread(imgpath,1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    img = cv2.resize(img, (90, 30))
    edge = cv2.Laplacian(img[:,:,0], cv2.CV_8U)
    edgemean = edge.mean()
    step = 10
    feat = []
    for y in range(0,img.shape[0],step):
        for x in range(0, img.shape[1], step):
            ys = range(y,y+step)
            xs = range(x,x+step)
            cy = img[ys,xs,0].mean()
            cu = img[ys,xs,1].mean()
            cv = img[ys,xs,2].mean()       
            ce = edge[ys,xs].mean() * 1.0 / edgemean
            feat.extend([cy,cu,cv,ce])
    return feat

def gen_batch_feat(imgdir):
    feats = []
    for root, dirs, names in os.walk(imgdir):
        for name in names:
            sname,ext = os.path.splitext(name)
            if 0 != cmp(ext, '.jpg') and 0 != cmp(ext, '.jpeg'):
                continue
            feat = gen_feat(os.path.join(root, name))
            feats.append(feat)
            if 0 == len(feats) % 1000:
                print len(feats), ':', name
    print len(feats), ' done! dim:', len(feats[0])
    return feats

def train(poss, negs, modelname):
    outsize = 2
    net = mlpbase.MLP_PROXY(modelname)
    posnum = len(poss)
    negnum = len(negs)
    samplenum = 2 * np.minimum(posnum,negnum)
    featdim = len(poss[0])
    samples = np.zeros((samplenum, featdim))
    targets = [[0] for k in range(samples.shape[0])]
    for k in range(samples.shape[0]/2):
        samples[k,:] = np.array(poss[k])
        targets[k][0] = 1
    for k in range(samples.shape[0]/2):
        idx = k + samples.shape[0]/2
        samples[idx,:] = np.array(negs[k])
        targets[idx][0] = 0

    targets = net.target_vec2mat(targets, outsize)
    samples, targets = net.shuffle(samples, targets)
    net.pre_normalization(samples)
    samples = net.normalization(samples)

    insize = samples.shape[1]

    net.create([insize, np.int64((insize+outsize)/3), outsize])

    print 'train :', samples.shape
    net.train(samples, targets, 1000, 0.001)
    net.save()
    print modelname, ' saved!'
    return

def predictK(samples, net):
    samples = net.normalization(samples)
    targets = net.predict(samples)
    targets = net.target_mat2vec(targets,2,-1) 
    res = []
    for k in range(len(targets)):
        res.append(targets[k][0])
    return res


def predict_images(indir, posdir,negdir, netname):
    net = mlpbase.MLP_PROXY(netname)
    net.load()
    posnum = 0
    negnum = 0
    for root, dirs, names in os.walk(indir):
        for name in names:
            sname,ext = os.path.splitext(name)
            ext.lower()
            if 0 != cmp(ext, '.jpg') and 0 != cmp(ext, '.jpeg'):
                continue
            feat = gen_feat(os.path.join(root, name))
            label = predictK(np.reshape(np.array(feat),(1,-1)),net)
            label = label[0]
            if label == 0:
                negnum += 1
                if negdir != None:
                    shutil.copy(os.path.join(root,name), negdir)        
            else:
                posnum += 1
                if posdir != None:
                    shutil.copy(os.path.join(root,name), posdir)       
            if (negnum + posnum) % 1000 == 0:
                print 'pos ratio = ', posnum * 1.0/(negnum  + posnum)

    print 'finished! pos ratio = ', posnum * 1.0/(negnum  + posnum)
    return

if __name__=="__main__":
    netpath = 'net.dat'
    if len(sys.argv) == 4 and 0==cmp(sys.argv[1], '-feat'):
        imgdir = sys.argv[2]
        outpath = sys.argv[3]
        feats = gen_batch_feat(imgdir)
        with open(outpath,'wb') as f:
            pickle.dump(feats, f)
        print len(feats) ,' samples saved in ', outpath
    if len(sys.argv) == 4 and 0 == cmp(sys.argv[1], '-train'):
        pospath = sys.argv[2]
        negpath = sys.argv[3]
        with open(pospath, 'rb') as f:
            poss = pickle.load(f)
        with open(negpath, 'rb') as f:
            negs = pickle.load(f)
        train(poss, negs, netpath)
    if len(sys.argv) == 3 and 0 == cmp(sys.argv[1], '-prdimg'):
        indir = sys.argv[2]
        predict_images(indir, None, None, netpath)

    if len(sys.argv) == 5 and 0 == cmp(sys.argv[1], '-prdimg'):
        indir = sys.argv[2]
        posdir = sys.argv[3]
        negdir = sys.argv[4]
        predict_images(indir, posdir, negdir, netpath)















