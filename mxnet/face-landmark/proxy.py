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

class FACENET_S2_PROXY(object):
    def __init__(self, outputNum, modelpath, cropSize, ctx, stdSize = 24):
        self.mean = [123.68, 116.28, 103.53]
        self.std = [58.395, 57.12, 57.375]
        self.outputNum = outputNum
        self.cropSize = cropSize
        self.stdSize = stdSize
        self.ctx= ctx
        mod = import_module('symbol.facenetS2')
        self.net = mod.get_symbol(outputNum,self.ctx)
        self.net.load_params(modelpath,ctx=self.ctx) 
        
    def predict(self, img, anchor,show_image = False):
        cx,cy = anchor
        x0 = cx - self.cropSize / 2
        y0 = cy - self.cropSize / 2
        if x0 < 0:
            x0 = 0
        if y0 < 0:
            y0 = 0
        x1 = x0 + self.cropSize
        y1 = y0 + self.cropSize
        if x1 >= img.shape[1]:
            x1 = img.shape[1]
        if y1 >= img.shape[0]:
            y1 = img.shape[0]
        if x1 - x0 < self.cropSize / 3 or y1 - y0 < self.cropSize / 3:
            return None
        x0,x1,y0,y1 = [ np.int64(x) for x in [x0,x1,y0,y1] ]
        X = cv2.resize( img[y0:y1,x0:x1,:], (self.stdSize, self.stdSize) )*1.0
        X = np.transpose(X,(2,0,1))
        
        for k in range(3):
            X[k,:,:] = (X[k,:,:] - self.mean[k]) / self.std[k]
        X = np.expand_dims(X,0) 
        X = mx.nd.array(X).as_in_context(self.ctx)
        Y = self.net(X)
        Y = np.reshape( Y.asnumpy(), (2,))
       # pdb.set_trace()
        Y[0] *= img.shape[1]
        Y[1] *= img.shape[0]
        if show_image:
            canvas = copy.deepcopy(img)
            x,y = np.int64(Y[0] + cx),np.int64( Y[1] + cy)
            cv2.circle(canvas, (x,y), 3, (255,0,0))
            cv2.imshow("predict",canvas)
            cv2.waitKey(-1)
        return Y


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
            #x += self.roi[0]
            #y += self.roi[2]
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
    NM1Proxy = FACENET_PROXY( (3,48,64), 6, "L1/NM1/weights.params", (0,64,16,64), ctx)
    F1Proxy = FACENET_PROXY( (3,64,64),10,"L1/F1/weights.params",(0,64,0,64), ctx )
    EN1Proxy = FACENET_PROXY( (3,48,64),6,"L1/EN1/weights.params",(0,64,0,48), ctx )
    LE21Proxy = FACENET_S2_PROXY( 2, "L2/LE21/weights.params",25, ctx, stdSize = 24)
    LE22Proxy = FACENET_S2_PROXY( 2, "L2/LE22/weights.params",30, ctx, stdSize = 24)
    RE21Proxy = FACENET_S2_PROXY( 2, "L2/RE21/weights.params",25, ctx, stdSize = 24)
    RE22Proxy = FACENET_S2_PROXY( 2, "L2/RE22/weights.params",30, ctx, stdSize = 24)    
    N21Proxy = FACENET_S2_PROXY( 2, "L2/N21/weights.params",25, ctx, stdSize = 24)
    N22Proxy = FACENET_S2_PROXY( 2, "L2/N22/weights.params",30, ctx, stdSize = 24)   

    LM21Proxy = FACENET_S2_PROXY( 2, "L2/LM21/weights.params",25, ctx, stdSize = 24)
    LM22Proxy = FACENET_S2_PROXY( 2, "L2/LM22/weights.params",30, ctx, stdSize = 24)     
    RM21Proxy = FACENET_S2_PROXY( 2, "L2/RM21/weights.params",25, ctx, stdSize = 24)
    RM22Proxy = FACENET_S2_PROXY( 2, "L2/RM22/weights.params",30, ctx, stdSize = 24)     
    #NM1Proxy.verify("c:/dataset/landmark/train/for-mxnet/NM1/", True)
    groundtruth = load_groundtruth( os.path.join( rootdir, 'landmarks.lst' ) )
    f1Error, f1Failure = [], []
    l1Error, l1Failure = [], []
    for jpg in os.listdir(rootdir):
        if '.jpg' != os.path.splitext(jpg)[-1]:
            continue
        img = cv2.imread(os.path.join(rootdir,jpg),1)
        img = cv2.resize(img,(stdSize, stdSize))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
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
        
        le21 = LE21Proxy.predict(img,(landmarks['left-eye'][0],landmarks['left-eye'][1]),False)
        le22 = LE22Proxy.predict(img,(landmarks['left-eye'][0],landmarks['left-eye'][1]),False)
        landmarks['left-eye'][0] += ( le21[0] + le22[0] ) * 0.5
        landmarks['left-eye'][1] += ( le21[1] + le22[1] ) * 0.5
     
        re21 = RE21Proxy.predict(img,(landmarks['right-eye'][0],landmarks['right-eye'][1]),False)
        re22 = RE22Proxy.predict(img,(landmarks['right-eye'][0],landmarks['right-eye'][1]),False)
        landmarks['right-eye'][0] += ( re21[0] + re22[0] ) * 0.5
        landmarks['right-eye'][1] += ( re21[1] + re22[1] ) * 0.5     
   
        n21 = N21Proxy.predict(img,(landmarks['nose'][0],landmarks['nose'][1]),False)
        n22 = N22Proxy.predict(img,(landmarks['nose'][0],landmarks['nose'][1]),False)
        landmarks['nose'][0] += ( n21[0] + n22[0] ) * 0.5
        landmarks['nose'][1] += ( n21[1] + n22[1] ) * 0.5 
        
        lm21 = LM21Proxy.predict(img,(landmarks['left-mouth'][0],landmarks['left-mouth'][1]),False)
        lm22 = LM22Proxy.predict(img,(landmarks['left-mouth'][0],landmarks['left-mouth'][1]),False)
        landmarks['left-mouth'][0] += ( lm21[0] + lm22[0] ) * 0.5
        landmarks['left-mouth'][1] += ( lm21[1] + lm22[1] ) * 0.5 
        
        rm21 = RM21Proxy.predict(img,(landmarks['right-mouth'][0],landmarks['right-mouth'][1]),False)
        rm22 = RM22Proxy.predict(img,(landmarks['right-mouth'][0],landmarks['right-mouth'][1]),False)
        landmarks['right-mouth'][0] += ( rm21[0] + rm22[0] ) * 0.5
        landmarks['right-mouth'][1] += ( rm21[1] + rm22[1] ) * 0.5 
        
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
        
        if show_image == True:
            for key in landmarks:
                x,y = [np.int64(k) for k in landmarks[key]]
                cv2.circle(img,(x,y),3,(0,255,255))
            cv2.imshow('res',img)
            cv2.waitKey(100)
    print "f1: (%f,%f), l1:(%f,%f)"%( np.mean( f1Error ),np.mean(f1Failure), np.mean(l1Error),np.mean(l1Failure))
        
if __name__== "__main__":
    main(sys.argv[1])
        
