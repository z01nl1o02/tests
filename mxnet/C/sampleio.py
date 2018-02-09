import mxnet as mx
import logging
import numpy as np
import gzip
import os,sys,pdb,pickle
import cv2

class SAMPLEIO(object):
    def __init__(self):
        return
    def load_one_class(self,indir,w,h):
        raws = []
        for jpg in os.listdir(indir):
            if os.path.splitext(jpg)[-1] != '.jpg':
                print 'skip ',jpg,' in ',indir
                continue
            fname = os.path.join(indir,jpg)
            img = cv2.imread(fname,1)
            img = cv2.resize(img,(w,h))
            img = np.float32(img) / 255.0 #normalization
            raw = [ img[:,:,0], img[:,:,1], img[:,:,2] ]
            raws.append(raw)
            if len(raws) > 3000:
                break
        datas = np.zeros( (len(raws), 3, w, h), dtype=np.float32)
        for row,raw in enumerate(raws):
            datas[row,0,:,:] = raw[0]
            datas[row,1,:,:] = raw[1]
            datas[row,2,:,:] = raw[2]
        return datas
        
    def load(self,indir,batchsize,w,h):
        dataall = []
        labelall = []
        for folder in os.listdir(indir):
            fname = os.path.join(indir,folder)
            if not os.path.isdir(fname):
                print 'skip ',folder
                continue
            #print folder
            datas = self.load_one_class(fname,w,h)
            dataall.append(datas)
            labelall.append( np.int32(folder) )
        total = reduce(lambda x,y: x + y, map(lambda z:z.shape[0], dataall))
        X = np.zeros( (total, 3, w, h), dtype=np.float32)
        Y = np.zeros( (total,), dtype=np.int32)
        row = 0
        for label,data in zip(labelall,dataall):
            X[row:row + data.shape[0],:,:,:] = data           
            Y[row:row + data.shape[0]] = label
            row += data.shape[0]
        #print X.shape, Y.shape
        self.data_iter = mx.io.NDArrayIter(X,Y,batchsize,shuffle=True)       
        return self
    def get_data_iter(self):
        return self.data_iter;


if __name__=="__main__":
    io = SAMPLEIO()
    io.load(sys.argv[1],32,32)

