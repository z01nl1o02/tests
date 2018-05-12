from mxnet import image
from mxnet import nd
import mxnet as mx
import pdb,cPickle
import utils,os,sys
from importlib import import_module
import numpy as np
import copy

#%matplotlib inline
import matplotlib as mpl
mpl.rcParams['figure.dpi']= 120
import matplotlib.pyplot as plt
import cv2


class FACENET_PROXY(object):
    def __init__(self, dataShape, outputNum, modelpath, roi, ctx, stdSize = 64):
        self.mean = [123.68, 116.28, 103.53]
        self.std = [58.395, 57.12, 57.375]
        self.dataShape = dataShape
        self.outputNum = outputNum
        self.roi = roi
        self.stdSize = stdSize
        self.ctx= ctx
        mod = import_module('symbol.facenet')
        self.net = mod.get_symbol(outputNum,self.ctx)
        self.net.load_params(modelpath,ctx=self.ctx)  

    def verify(self, path_root, show_image=False):
        aug = mx.image.CreateAugmenter(data_shape = self.dataShape, resize = self.stdSize , mean=True, std=True)
        dataIter = mx.image.ImageIter(batch_size=1, data_shape=self.dataShape, 
                label_width = self.outputNum,
                path_imglist=os.path.join(path_root,'landmarks.lst'),
                path_root=path_root,
                aug_list=aug)      
        n = 0
        acc = nd.array([0])
        for batch in dataIter:
            #pdb.set_trace()
            data, label, batch_size = utils._get_batch(batch, [self.ctx])
            for X, y in zip(data, label):
                y = y.astype('float32')
                y0 = self.net(X)
                acc += nd.sum( (y0-y)*(y0-y) ).copyto(mx.cpu())
                n += y.shape[0]
                if show_image:
                    img = X.asnumpy()[0]
                    for k in range(3):
                        img[k,:,:] = img[k,:,:] * self.std[k] + self.mean[k] #restore mean/std                     
                    img = np.transpose(img,(1,2,0))
                    img = cv2.cvtColor( np.uint8(img), cv2.COLOR_BGR2GRAY)
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                    img = cv2.resize(img, (img.shape[1] * 1, img.shape[0]*1))
                    for k in range(0,self.outputNum,2):
                        x,y = y0.asnumpy()[0,k],y0.asnumpy()[0,k+1]
                        x,y = np.int64(x * img.shape[1]), np.int64(y * img.shape[0])
                        cv2.circle(img, (x,y), 3,(128,255,128))                     
                    cv2.imshow("src",np.uint8(img))
                    cv2.waitKey(-1)
            acc.wait_to_read()
        print 'total ',n
        print 'acc = ',acc.asscalar() / (2*n)
    def predict(self, img, show_image = False):
        x0,x1,y0,y1 = self.roi
        
        X = img[y0:y1,x0:x1,:]*1.0
        X = np.transpose(X,(2,0,1))
        
        for k in range(3):
            X[k,:,:] = (X[k,:,:] - self.mean[k]) / self.std[k]
        X = np.expand_dims(X,0) 
        X = mx.nd.array(X).as_in_context(self.ctx)
        Y = self.net(X)
        Y = Y.asnumpy()
        for k in range(0, Y.shape[1], 2):
            x,y = Y[0,k] * self.stdSize, Y[0,k+1]*self.stdSize
            x += self.roi[0]
            y += self.roi[2]
            Y[0,k], Y[0,k+1] = x,y
        if show_image:
            canvas = copy.deepcopy(img)
            for k in range(0, Y.shape[1], 2):
                x,y = Y[0,k],Y[0,k+1]
                cv2.circle(canvas, (x,y), 3, (255,0,0))
            cv2.imshow("predict",canvas)
            cv2.waitKey(-1)
        return Y

def bagging_mean(landmarks):
    res = {}
    for key in landmarks.keys():
        res[key] = [0,0]
        for k in range( len(landmarks[key]) ):
            x,y = landmarks[key][k]
            res[key][0] += x
            res[key][1] += y
        res[key][0] /= len(landmarks[key])
        res[key][1] /= len(landmarks[key])
    return res
            
def load_groundtruth(filepath):
    groundtruth = {}
    with open(filepath, 'rb') as f:
        for line in f:
            line = line.strip()
            if line == "":
                continue
            line = line.split('\t')
            path = line[-1]
            pts = [np.float64(x) for x in line[1:-1]]            
            groundtruth[ path ] = np.reshape( np.asarray(pts),(-1,2) )
    print('load groundtruth %d'%(len(groundtruth)))
    return groundtruth
    
def main(rootdir, stdSize = 64, show_image=False):
    ctx = mx.gpu()
    NM1Proxy = FACENET_PROXY( (3,48,64), 6, "NM1/epoch-000004.params", (0,64,16,64), ctx)
    F1Proxy = FACENET_PROXY( (3,64,64),10,"F1/epoch-000008.params",(0,64,0,64), ctx )
    EN1Proxy = FACENET_PROXY( (3,48,64),6,"EN1/epoch-000009.params",(0,64,0,48), ctx )
    #NM1Proxy.verify("c:/dataset/landmark/train/for-mxnet/NM1/", True)
    groundtruth = load_groundtruth( os.path.join( rootdir, 'landmarks.lst' ) )
    f1Error, f1Failure = [], []
    l1Error, l1Failure = [], []
    for jpg in os.listdir(rootdir):
        if '.jpg' != os.path.splitext(jpg)[-1]:
            continue
        img = cv2.imread(os.path.join(rootdir,jpg),1)
        img = cv2.resize(img,(stdSize, stdSize))
        nm1 = NM1Proxy.predict(img, False)
        f1 = F1Proxy.predict(img,False)
        en1 = EN1Proxy.predict(img,False)
        
        landmarks = {}
        landmarks['left-eye'] = ( (f1[0,0],f1[0,1]), (en1[0,0],en1[0,1]) )
        landmarks['right-eye'] = ( (f1[0,2],f1[0,3]), (en1[0,2],en1[0,3]) )
        landmarks['nose'] = ( (f1[0,4],f1[0,5]), (en1[0,4],en1[0,5]), (nm1[0,0],nm1[0,1]) )
        landmarks['left-mouth'] = ( (f1[0,6],f1[0,7]), (nm1[0,2],nm1[0,3]) )
        landmarks['right-mouth'] = ( (f1[0,8],f1[0,9]), (nm1[0,4],nm1[0,5]) )
        landmarks =bagging_mean(landmarks)
        #@pdb.set_trace()
        if jpg in groundtruth.keys():
            f1Res = np.zeros( ( f1.shape[1]/2, 2) )
            l1Res = np.zeros( ( f1.shape[1]/2, 2) )
            for k in range(0, f1.shape[1], 2):
                x,y = f1[0,k]/stdSize, f1[0,k+1]/stdSize
                f1Res[k/2][0], f1Res[k/2][1] = x, y
                
            for k,key in enumerate( ['left-eye','right-eye','nose','left-mouth','right-mouth'] ):
                x,y = landmarks[key][0]/stdSize,landmarks[key][1]/stdSize
                l1Res[k,0], l1Res[k,1] = x,y
            
            gnd = groundtruth[jpg]    
            err =  np.sqrt( (( f1Res - gnd ) ** 2).sum(axis=1) )
            f1Error.extend(  err.tolist() )
            f1Failure.append( (err > 0.05).tolist() )
            
            err =  np.sqrt( (( l1Res - gnd ) ** 2).sum(axis=1) )
            l1Error.append( err.tolist() )
            l1Failure.append( (err > 0.05).tolist() )  
        f1Failure = [np.float64(x) for x in f1Failure]
        l1Failure = [np.float64(x) for x in l1Failure]
        print "f1: (%f,%f), l1:(%f,%f)"%( np.mean( f1Error ),np.mean(f1Failure), np.mean(l1Error),np.mean(l1Failure))
        if show_image == True:
            for key in landmarks:
                x,y = [np.int64(k) for k in landmarks[key]]
                cv2.circle(img,(x,y),3,(0,255,255))
            cv2.imshow('res',img)
            cv2.waitKey(100)

        
if __name__== "__main__":
    main(sys.argv[1])
        
