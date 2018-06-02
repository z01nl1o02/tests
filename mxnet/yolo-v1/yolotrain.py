import os,sys,pdb
import numpy as np
import mxnet as mx

import iterator
from mxnet.gluon.model_zoo import vision
from mxnet.gluon import nn
import mxnet.ndarray as nd
from mxnet import gluon
import time
from mxnet import autograd
import math

from mxnet import image


trainRecPath = 'data/train.rec'
batchSize = 8
dataShape = 256
numClasses = 2
box_per_cell = 2
mean_red, mean_blue, mean_green = 123, 117, 104
rgb_mean = nd.array([123, 117, 104])
rgb_std = nd.array([58.395, 57.12, 57.375])
maxEpoch = 200

data_dir = 'data\\'

#1--create data iter
#trainIter = iterator.DetRecordIter(trainRecPath,batchSize,(3, dataShape, dataShape),\
#                                  mean_pixels=(mean_red, mean_green, mean_blue), std_pixel = ( rgb_std[0], rgb_std[1], rgb_std[2]  ))
                                      
trainIter = image.ImageDetIter(
        batch_size=batchSize,
        data_shape=(3, dataShape, dataShape),
        path_imgrec=data_dir+'train.rec',
        path_imgidx=data_dir+'train.idx',
        shuffle=True,
        mean=True,
        std=True,
        rand_crop=1,
        min_object_covered=0.95,
        max_attempts=200)
        
batch = trainIter.next()
label, data = batch.label, batch.data
print(label[0].shape,"batchSize, maxObjectNum,classId,xmin,ymin,xmax,ymax,difficult")
print(data[0].shape, "batchSize,C,H,W")

#2--load net symbol
#2.1-load pretrained net (feature part)
pretrained = vision.get_model('resnet18_v1', pretrained=True).features
net = nn.HybridSequential()
for i in range(len(pretrained) - 2):
    net.add(pretrained[i])
#not to initialize the pretrained model???


#2.2-add yolo output
class YOLOV1_OUTPUT(nn.HybridBlock):
    def __init__(self, numClass, box_per_cell,verbose = False, **kwargs):
        super(YOLOV1_OUTPUT, self).__init__(**kwargs)
        self.numClass = numClass
        self.box_per_cell = box_per_cell
        channelNum = box_per_cell * (numClass  + 5) #IOU,x,y,w,h
        #(x,y) is related to cell
        #(w,h) is related to (W,H) of last feature
        with self.name_scope():
            self.conv = nn.Conv2D(channelNum, 1,1)
        self.verbose = verbose
        return
    def forward(self, x, *args):
        y = self.conv(x)
        if self.verbose:
            print('yolo_output:',x.shape, '->', y.shape)
        return y
yolov1_output =  YOLOV1_OUTPUT(numClasses,box_per_cell,verbose=False)
yolov1_output.initialize()
net.add(yolov1_output)
#check data shape in final layer
#X = nd.random_normal(0,1,shape=(batchSize, 3, dataShape, dataShape))
#Y = net.forward(X)

#3--define loss


#4--get trainer
ctx = mx.gpu()
net.collect_params().reset_ctx(ctx)
trainer = gluon.Trainer(net.collect_params(),"sgd",{"learning_rate":1,"wd":5e-4})



#5--start train
#5.1-parse net output
def convert_xy(XY):
    B,H,W,A,N = XY.shape
    dy = nd.tile( nd.arange(0,H,repeat=(W*A), ctx = XY.context).reshape((1,H,W,A,1)), (B,1,1,1,1) )
    dx = nd.tile( nd.arange(0,W,repeat=(A),ctx = XY.context).reshape((1,1,W,A,1)), (B,H,1,1,1) )
    x,y = XY.split(num_outputs=2,axis=-1)
    x = (x + dx) / W
    y = (y + dy) / H
    return x,y
def convert_wh(WH):
    B,H,W,A,N = WH.shape
    w,h = WH.split(num_outputs=2,axis=-1)
    return w,h
def parse_net_output(Y,numClass, box_per_cell):
    pred = nd.transpose(Y,(0,2,3,1))
    pred = pred.reshape((0,0,0,box_per_cell,numClass + 5)) #add one dim for boxes
    predCls = nd.slice_axis(pred, begin = 0, end = numClass,axis=-1)
    predObject = nd.slice_axis(pred,begin=numClass,end=numClass+1,axis=-1)
    #predObject = nd.sigmoid(predObject)
    predXY = nd.slice_axis(pred, begin = numClass + 1, end = numClass + 3, axis=-1)
    #predXY = nd.sigmoid(predXY)
    predWH = nd.slice_axis(pred, begin = numClass + 3, end = numClass + 5, axis=-1)
    #x,y = convert_xy(predXY)
    #w,h = convert_wh(predWH)
    #w = nd.clip(w,0,1)
    #h = nd.clip(h,0,1)
    #x0 = nd.clip(x, 0, 1)
    #y0 = nd.clip(y,0,1)
    #x1 = nd.clip(x0 + w,0,1)
    #y1 = np.clip(y0 + h, 0,1)
    #x = x0
    #y = y0
    #w = x1 - x0
    #h = y1 - y0
    XYWH = nd.concat(predXY,predWH,dim=-1)
   # pdb.set_trace()
    return predCls, predObject, XYWH

def parse_groundtruth_for_target(labels, box_per_cell, xywh):
    B,H,W,A,_ = xywh.shape
    _,maxObjNum,_ = labels.shape
    #pdb.set_trace()
    boxMask = nd.zeros( (B,H,W,A,1), ctx = xywh.context )
    boxCls = nd.ones_like(boxMask, ctx = xywh.context) * (-1) #default negative label
    boxObject = nd.zeros((B,H,W,A,1),ctx = xywh.context)
    boxXYWH = nd.zeros((B,H,W,A,4), ctx = xywh.context)
    for b in range(B):
        label  = labels[b].asnumpy()
        validLabel = label[np.where(label[:,1] >-0.5)[0],:]
        #pdb.set_trace()
        np.random.shuffle(validLabel)
        for l in validLabel:
            cls,x0,y0,x1,y1 = l
            w = x1 - x0
            h = y1 - y0
            #find best box for this object
            indx,indy = int(x0*W), int(y0*H) #position
            pws, phs = xywh[b,indy, indx, :, -2], xywh[b,indy,indx,:,-1]
            ious = []
            pws = pws.asnumpy()
            phs = phs.asnumpy()
            pws, phs = [1,1],[1,1]
            
            for pw, ph in zip(pws,phs):
                intersect = np.minimum(pw,w*W) * np.minimum(ph,h*H)
                ious.append(  intersect / (pw * ph + w * h - intersect) )
            #pdb.set_trace()
            bestbox = int(np.argmax(ious))
            boxMask[b,indy,indx,bestbox,:] = 1.0
            boxCls[b,indy,indx,bestbox,:] = cls
            boxObject[b,indy,indx,bestbox,:] = 1.0 # ious[bestbox]
            tx = x0 * W - indx
            ty = y0 * H - indy
            tw,th = math.sqrt(w),  math.sqrt(h) #predict sqrt(w) sqrt(h)
            #pdb.set_trace()
            boxXYWH[b,indy,indx,bestbox,:] = nd.array([tx,ty,tw,th])
    return boxMask, boxCls, boxObject,boxXYWH

#5.2 loss
loss_sce = gluon.loss.SoftmaxCrossEntropyLoss(from_logits=False)
loss_l1 = gluon.loss.L1Loss()
class LOSS_RECORDER(mx.metric.EvalMetric):
    def __init__(self, name):
        super(LOSS_RECORDER,self).__init__(name)
    def update(self, labels, pred = 0):
        for loss in labels:
            if isinstance(loss, mx.nd.NDArray):
                loss = loss.asnumpy()
            self.sum_metric += loss.sum()
            self.num_inst += 1
        return

obj_loss = LOSS_RECORDER('objectness_loss')
cls_loss = LOSS_RECORDER('classes_loss')
xywh_loss = LOSS_RECORDER('xywh_loss')

positive_weight = 5.0
negative_weight = 0.1
class_weight = 1.0
xywh_weight = 5.0

for epoch in range(maxEpoch):
    trainIter.reset()
    tic = time.time()
    for batchidx, batch in enumerate(trainIter):
        Y0 = batch.label[0].as_in_context(ctx)
        X = batch.data[0].as_in_context(ctx)
        with autograd.record():
            Y1 = net(X)
            predCls, predObj, predXYWH = parse_net_output(Y1,numClasses, box_per_cell)
            with autograd.pause(): #generate ground online
                boxMask, boxCls, boxObj, boxXYWH = parse_groundtruth_for_target(Y0,box_per_cell,predXYWH)
            if 0:
                lines = []
                for y in range(16):
                    for x in range(16):
                        a = boxMask[0,y,x,0,0].asnumpy()[0]
                        b = boxMask[0,y,x,1,0].asnumpy()[0]
                        c = '-'
                        #pdb.set_trace()
                        if a > 0.5:
                            c = boxXYWH[0,y,x,0,:].asnumpy().tolist()
                            c = ['%.2f'%cc for cc in c]
                            c = '-'.join(c)
                        elif b > 0.5:
                            c = boxXYWH[0,y,x,1,:].asnumpy().tolist()
                            c = ['%.2f'%cc for cc in c]
                            c = '-'.join(c)                            
                        lines.append('[%.3f,%.3f,%s],'%(a,b,c))
                    lines.append('')
                with open('dbg.txt','wb') as f:
                    f.write('\r\n'.join(lines))
                #pdb.set_trace()
            loss0 = loss_sce(predCls, boxCls, boxMask * class_weight)
            boxWeight = nd.where( boxMask > 0, boxMask * positive_weight, boxMask * negative_weight  )
            loss1 = loss_l1(predObj, boxObj, boxWeight)
            #pdb.set_trace()
            loss2 = loss_l1(predXYWH, boxXYWH, xywh_weight * boxMask  )
            loss = loss0 + loss1 + loss2
        loss.backward()
        trainer.step(batchSize)
        cls_loss.update(loss0)
        obj_loss.update(loss1)
        xywh_loss.update(loss2)
    print '%d,(%s,%f),(%s,%f),(%s,%f),%f'%(epoch,cls_loss.get()[0], cls_loss.get()[1],\
        obj_loss.get()[0], obj_loss.get()[1], xywh_loss.get()[0], xywh_loss.get()[1], time.time() - tic)
    net.save_params('%d.params'%epoch)



