import numpy as np
import mxnet as mx
import mxnet.gluon as gluon
import mxnet.gluon.nn as nn
import mxnet.autograd as autograd
import os,sys,pdb

root='c:/dataset/cifar/split/'
outdir = 'output/'
pretrain = -1 #round number


batchSize=20
imgSize=28 #after crop
channelNum=3
classNum = 10
dataShape=(batchSize,channelNum,imgSize, imgSize)
ctx = mx.gpu()

trainAugList = mx.image.CreateAugmenter((channelNum,imgSize, imgSize),rand_crop=True,rand_mirror=True,mean=True,std=True)
trainIter = mx.image.ImageIter(batchSize,(channelNum,imgSize, imgSize),label_width=1,
                               path_imglist='train.lst',path_root=os.path.join(root,'train'),
                               shuffle=True,aug_list=trainAugList)

testAugList = mx.image.CreateAugmenter((channelNum,imgSize, imgSize),rand_crop=False,mean=True,std=True)
testIter = mx.image.ImageIter(batchSize,(channelNum,imgSize, imgSize),label_width=1,
                               path_imglist='test.lst',path_root=os.path.join(root,'test'),
                               shuffle=False,aug_list=testAugList)

class CIFARCONV(nn.HybridBlock):
    def __init__(self,ch,kernel=3,stride=1,padding=1,**kwargs):
        super(CIFARCONV, self).__init__(**kwargs)
        with self.name_scope():
            self.conv = nn.Conv2D(channels=ch, kernel_size=kernel, strides=stride,padding=padding)
            self.bn=nn.BatchNorm()
        return
    def hybrid_forward(self, F, x, *args, **kwargs):
        out=F.relu(self.bn(self.conv(x)))
        return out

class CIFARNET(nn.HybridBlock):
    def __init__(self,classNum,verbose=False,**kwargs):
        super(CIFARNET,self).__init__(**kwargs)
        with self.name_scope():
            self.conv1=CIFARCONV(ch=6,stride=1,kernel=3,padding=1)
            self.conv2=CIFARCONV(ch=6,stride=1,kernel=5,padding=2)
            self.conv3=CIFARCONV(ch=12,stride=2,kernel=3,padding=1)
            self.conv4=CIFARCONV(ch=24,stride=2,kernel=3,padding=1)
            self.pool = nn.GlobalAvgPool2D()
            self.fc1 = nn.Dense(64)
            self.fc2 = nn.Dense(classNum)
        return
    def hybrid_forward(self, F, x, *args, **kwargs):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = F.relu( self.fc1( self.pool(out) ) )
        out = F.relu( self.fc2(out))
        return out

net = CIFARNET(classNum)
net.initialize(ctx = ctx)
net.hybridize()

if pretrain >= 0:
    net.load_params(os.path.join(outdir,'cifar-%.4d.params'%pretrain),ctx=ctx)
    print 'load model'


trainer = gluon.Trainer(net.collect_params(), "sgd", {'learning_rate':1.0,"wd":0.00005})

loss_ce = gluon.loss.SoftmaxCrossEntropyLoss()

class LOSSREC(mx.metric.EvalMetric):
    def __init__(self,name):
        super(LOSSREC,self).__init__(name)
        return
    def update(self, labels, preds = 0):
        for loss in labels:
            if isinstance(loss, mx.nd.NDArray):
                loss = loss.asnumpy()
            self.sum_metric += loss.sum()
            self.num_inst += 1
        return

train_loss = LOSSREC("train-error")
test_loss = LOSSREC("test-error")

def calc_hr(predY,Y):
    if isinstance(predY,mx.nd.NDArray):
        predY = predY.asnumpy()
    if isinstance(Y,mx.nd.NDArray):
        Y = Y.asnumpy()
    num, dim = predY.shape
    rowMax = np.tile( np.reshape( predY.max(axis=1), (-1,1) ), (1, dim ) )
    res = rowMax == predY
    Y = np.reshape( Y,(1,-1) )
    hr = np.mean( [ res[row,np.int32(col)] for row, col in enumerate(Y.tolist()[0])  ] )
    return hr

from matplotlib import pyplot as plt

class VISUAL_LOSS(object):
    def __init__(self):
        plt.ion()
        self.trainloss = []
        self.testloss = []
        return
    def reduce(self,th = 100):
        if len(self.trainloss) > th:
            self.trainloss = self.trainloss[th//10:]
        if len(self.testloss) > th:
            self.testloss = self.testloss[th//10:]
        return
    def update_train(self, round, loss):
        if isinstance(loss, mx.nd.NDArray):
            loss = loss.asnumpy()[0]
        self.trainloss.append((round, loss))
        return
    def update_test(self,round,loss):
        if isinstance(loss, mx.nd.NDArray):
            loss = loss.asnumpy()[0]
        self.testloss.append((round,loss))
    def show(self):
        self.reduce()
        if len(self.trainloss) > 0:
            x = [d[0] for d in self.trainloss]
            y = [d[1] for d in self.trainloss]
            plt.plot(x,y,"r")
        if len(self.testloss) > 0:
            x = [d[0] for d in self.testloss]
            y = [d[1] for d in self.testloss]
            plt.plot(x,y,"b")
        plt.pause(0.05)
        return

from time import time
t0 = time()

visualloss = VISUAL_LOSS()

lr_steps = [5000,10000,20000,40000]

round = 0
for epoch in range(200):
    trainIter.reset()
    for batchidx, batch in enumerate(trainIter):
        round += 1
        if round in set(lr_steps):
            trainer.set_learning_rate(trainer.learning_rate * 0.1)


        X,Y = batch.data[0].as_in_context(ctx), batch.label[0].as_in_context(ctx)
        with autograd.record():
            predY = net.forward(X)
            loss = loss_ce(predY,Y)
        loss.backward()
        trainer.step(batchSize)
        train_loss.update(loss)
        if batchidx % 100 == 0:
            print 'round {} {}'.format(round,train_loss.get())
            visualloss.update_train(round,train_loss.get()[1])
            visualloss.show()
    hrlist = []
    testIter.reset()
    for batch in testIter:
        X,Y = batch.data[0].as_in_context(ctx), batch.label[0].as_in_context(ctx)
        predY = net.forward(X)
        loss = loss_ce(predY,Y)
        test_loss.update(loss)
        hrlist.append(calc_hr(predY,Y))
    visualloss.update_test(round,test_loss.get()[1])
    visualloss.show()
    hr = np.mean(hrlist)
    print 'epoch {} {:.2f} min {} {} hr:{:.2f}'.format( epoch, (time()-t0)/60.0,
              train_loss.get(),test_loss.get(), hr)
    #net.export(os.path.join(outdir,"cifar"),epoch=round)
    net.save_params(os.path.join(outdir,'cifar-%.4d.params'%round))

plt.show()


