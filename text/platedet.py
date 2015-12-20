import os,sys,pdb,pickle,cv2,shutil
import numpy as np
import mlpbase
import multiprocessing as mp

def gen_featK(img):
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
            ce = edge[ys,xs].mean() * 1.0 / (edgemean + 0.01)
            feat.extend([cy,cu,cv,ce])
    return feat

 
def gen_feat(imgpath):
    img = cv2.imread(imgpath,1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    return gen_featK(img)

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

def predict_slidingwindowK(imgpath, netname, outpath):
    net = mlpbase.MLP_PROXY(netname)
    net.load()
    img = cv2.imread(imgpath, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    objs = []
    sizelist = []
    lastw = 90
    for k in range(5):
        sizelist.append(np.int64(lastw))
        lastw = lastw * 1.2

    for objw in sizelist:
        objh = np.int64(objw / 3) 
        if objh < 10:
            continue
        stepw = objw / 5
        steph = objh / 5
        if stepw < 5:
            stepw = 5
        if steph < 5:
            steph = 5
        for y in range(0, img.shape[0] - objh, steph):
            for x in range(0, img.shape[1] - objw, stepw):
                subimg = img[y:y+objh, x:x+objw, :]
                feat = gen_featK(subimg)
                label = predictK(np.reshape(np.array(feat),(1,-1)), net)[0]
                if label == 1:
                    objs.append([x,y, x+objw, y + objh])
    if outpath != None:
        img = cv2.cvtColor(img, cv2.COLOR_YUV2BGR)
        for rect in objs:
            x0,y0,x1,y1 = rect
            cv2.rectangle(img, (x0,y0),(x1,y1), (255,0,0),2)
        cv2.imwrite(outpath, img)
    return objs

def predict_slidingwindow(indir, outdir, netname):
    pool = mp.Pool(3)
    for root, dirs, names in os.walk(indir):
        for name in names:
            sname,ext = os.path.splitext(name)
            ext.lower()
            if 0 != cmp(ext, '.jpg') and 0 != cmp(ext,'.jpeg'):
                continue
            src = os.path.join(root, name)
            dst = os.path.join(outdir, name)
            pool.apply_async(predict_slidingwindowK,(src, netname, dst))
    pool.close()
    pool.join()
    return 

def save_feats(feats, outdir):
    step = 10000
    idx = 0
    for k in range(0,len(feats), step):
        x0 = k
        x1 = k + step
        with open(os.path.join(outdir, str(idx) + '.dat'), 'wb') as f:
            pickle.dump(feats[x0:x1], f)
        idx += 1
    return  
  
def load_feats(indir):
    feats = []
    for root, dirs, names in os.walk(indir):
        for name in names:
            sname,ext = os.path.splitext(name)
            if 0 != cmp(ext, '.dat'):
                continue
            with open(os.path.join(indir, name), 'rb') as f:
                fs = pickle.load(f)
            feats.extend(fs)
    return feats

    
if __name__=="__main__":
    netpath = 'net.dat'
    if len(sys.argv) == 4 and 0==cmp(sys.argv[1], '-feat'):
        imgdir = sys.argv[2]
        outdir = sys.argv[3]
        feats = gen_batch_feat(imgdir)
        save_feats(feats, outdir)
        print len(feats) ,' samples saved in ', outdir
    if len(sys.argv) == 4 and 0 == cmp(sys.argv[1], '-train'):
        posdir = sys.argv[2]
        negdir = sys.argv[3]
        poss = load_feats(posdir)
        negs = load_feats(negdir)
        train(poss, negs, netpath)
    if len(sys.argv) == 3 and 0 == cmp(sys.argv[1], '-prdimg'):
        indir = sys.argv[2]
        predict_images(indir, None, None, netpath)
    if len(sys.argv) == 5 and 0 == cmp(sys.argv[1], '-prdimg'):
        indir = sys.argv[2]
        posdir = sys.argv[3]
        negdir = sys.argv[4]
        predict_images(indir, posdir, negdir, netpath)
    if len(sys.argv) == 4 and 0 == cmp(sys.argv[1], '-prdsw'):
        indir = sys.argv[2]
        outdir = sys.argv[3]
        predict_slidingwindow(indir,outdir,netpath)














