import numpy as np
import mxnet as mx
import mxnet.ndarray as nd
from mxnet import gluon
import cv2,pickle
import gzip,random
from mxnet import autograd
from time import time
batchSize = 100
dataShape = (batchSize,1,28,28)
ctx = mx.cpu()
verbose = False

class DSITER_FAST(mx.io.DataIter):
    def __init__(self,dataname,batchSize,datafile='mnist.pkl.gz'):
        with gzip.open(datafile,'rb') as f:
            trainset,validset,testset = pickle.load(f)
        self.X = None
        self.Y = None
        self.shuffle = False
        if dataname == "train":
            self.X = trainset[0]
            self.Y = trainset[1]
            self.shuffle = True
        elif dataname == 'valid':
            self.X = validset[0]
            self.Y = validset[1]
            self.shuffle = False
        elif dataname == 'test':
            self.X = testset[0]
            self.Y = testset[1]
            self.shuffle = False
        else:
            print("unk data name %s"%dataname)
            return
        self.iter = mx.io.NDArrayIter(self.X,self.Y,batchSize,self.shuffle)
        return
    def __iter__(self):
        return self
    def __next__(self):
        return self.next()
    def reset(self):
        self.iter.reset()
        return
    def next(self):
        return self.iter.next()

class DSITER(mx.io.DataIter):
    def __init__(self,dataname,shape,datafile='mnist.pkl.gz'):
        with gzip.open(datafile,'rb') as f:
            trainset,validset,testset = pickle.load(f)
        self.X = None
        self.Y = None
        self.nextIdx = 0
        self.indexes = []
        self.shape = shape #Batch, C, H, W
        if dataname == "train":
            self.X = trainset[0]
            self.Y = trainset[1]
            self.fortrain = True
        elif dataname == 'valid':
            self.X = validset[0]
            self.Y = validset[1]
            self.fortrain = False
        elif dataname == 'test':
            self.X = testset[0]
            self.Y = testset[1]
            self.fortrain = False
        else:
            print("unk data name %s"%dataname)
            return
        self.total = len(self.X)
        self.indexes = [k for k in range(self.total)]
        random.shuffle(self.indexes)
        return
    def __iter__(self):
        return self
    def __next__(self):
        return self.next()
    def reset(self):
        self.nextIdx = 0
        random.shuffle(self.indexes)
        return
    def get_image(self,X):
        B,C,H,W = self.shape
        X = np.reshape(X,(28,28))
        X = X[:,:,np.newaxis]
        X = np.tile(X,(1,1,3))
        if H > X.shape[0] or W > X.shape[1]:
            raise RuntimeError
        if H < X.shape[0] or W < X.shape[1]:
            if self.fortrain:
                X, _ = mx.image.random_crop(nd.array(X),(H,W))
            else:
                X,_ = mx.image.center_crop(nd.array(X),(H,W))
            X = np.transpose(X.asnumpy(),(2,0,1))
        else:
            #print "data augment is off"
            X = np.transpose(X,(2,0,1))
        return X
    def next(self):
        if self.nextIdx + self.shape[0] > self.total:
            raise StopIteration
        p0,p1 = self.nextIdx, self.nextIdx + self.shape[0]
        X = [self.get_image( self.X[self.indexes[k]] ) for k in range(p0,p1)]
        Y = [self.Y[self.indexes[k]] for k in range(p0,p1)]
        X = np.array(X)
        Y = np.array(Y)
        self.nextIdx += self.shape[0]
        return mx.io.DataBatch(data = [X], label = [Y])
    @property
    def batch_shape(self):
        return self.shape


class OCRNET(gluon.HybridBlock):
    def __init__(self,numClass,verbose=False,**kwargs):
        super(OCRNET,self).__init__(**kwargs)
        self.verbose = verbose
        with self.name_scope():
            #self.conv1 = gluon.nn.Conv2D(6,kernel_size=(3,3),strides=(1,1),padding=(1,1))
            #self.pool1 = gluon.nn.MaxPool2D(pool_size=(2,2))
            #self.conv2 = gluon.nn.Conv2D(12,kernel_size=(3,3),strides=(1,1),padding=(1,1))
            #self.pool2 = gluon.nn.MaxPool2D(pool_size=(2,2))
            #self.pool3 = gluon.nn.GlobalAvgPool2D()
            #self.fc4 = gluon.nn.Dense(numClass)
            self.flatten = gluon.nn.Flatten()
            self.fc1 = gluon.nn.Dense(128)
            self.fc2 = gluon.nn.Dense(64)
            self.fc3 = gluon.nn.Dense(numClass)
            self.act3 = gluon.nn.Activation('sigmoid')
        return
    def hybrid_forward(self,F,x):
        if self.verbose:
            print 'OCRNET input: ',x.shape
        x = self.flatten(x)
        #x = self.pool1( F.relu(self.conv1(x)) )
        if self.verbose:
            print 'OCRNET stage 1: ',x.shape
        #x = self.pool2( F.relu(self.conv2(x)) )
        x = F.relu( self.fc1(x) )
        if self.verbose:
            print 'OCRNET stage 2: ',x.shape
        #x = self.pool3(x)
        x = F.relu( self.fc2(x) )
        if self.verbose:
            print 'OCRNET stage 3: ',x.shape
        #x = self.fc4(x)
        x = self.act3(self.fc3(x))
        return x


class LOSS_EVAL(mx.metric.EvalMetric):
    def __init__(self,name):
        super(LOSS_EVAL, self).__init__(name)
    def update(self,losses,pred=0):
        for loss in losses:
            if isinstance(loss,mx.nd.NDArray):
                loss = loss.asnumpy()
            self.sum_metric  += loss.sum()
            self.num_inst += 1
        return

trainIter = DSITER("train",dataShape)
validIter = DSITER("valid",dataShape)


#trainIter = DSITER_FAST("train",batchSize)
#validIter = DSITER_FAST("valid",batchSize)


if verbose:
    for batchData in trainIter:
        X = batchData.label[0]
        Y = batchData.data[0]
        print X.shape, Y.shape
        break

net = OCRNET(10,verbose=verbose)
net.initialize(ctx=ctx)
net.hybridize()
if verbose:
    print 'show net shape'
    X = np.random.rand(dataShape[0],dataShape[1],dataShape[2],dataShape[3])
    X = nd.array(X)
    Y = net(X)
    print Y.shape

loss_ce = mx.gluon.loss.SoftmaxCrossEntropyLoss()
trainer = mx.gluon.Trainer(net.collect_params(),"sgd",{"learning_rate":0.1,"wd":5e-4})

cls_loss = LOSS_EVAL("cls")

for epoch in range(100):
    #print('start epoch %d'%epoch)
    t0 = time()
    trainIter.reset()
    for batchIdx, batchData in enumerate(trainIter):
        Y = nd.array( batchData.label[0] ).as_in_context(ctx)
        X = nd.array( batchData.data[0] ).as_in_context(ctx)
        with autograd.record():
            predY = net(X)
            loss = loss_ce(predY,Y)
        loss.backward()
        trainer.step(batchSize)
        cls_loss.update(loss)
    validIter.reset()
    num = 0
    loss = 0
    hit = 0
    for batchIdx, batchData in enumerate(validIter):
        Y = nd.array( batchData.label[0] ).as_in_context(ctx)
        X = nd.array( batchData.data[0] ).as_in_context(ctx)
        predY = net(X)
        loss += loss_ce(predY,Y).asnumpy().sum()
        Y,predY = Y.asnumpy(), predY.asnumpy()
        Y = np.reshape(Y,(batchSize,1))
        predY = np.reshape(predY,(batchSize,10))
        for idx in range(batchSize):
            if predY[idx,:].max() == predY[idx,np.int32(Y[idx])]:
                hit += 1
        num += 1
    print('epoch %d,%.2fms, train loss %f test loss %f test precision %f'%(epoch,time() - t0,cls_loss.get()[1],  \
          loss / (num * batchSize), hit * 1.0 / (num * batchSize) ))
    net.save_params('ocrnet_%.5d.params'%epoch)
