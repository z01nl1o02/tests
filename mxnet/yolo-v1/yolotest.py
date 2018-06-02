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


trainRecPath = 'train.rec'
batchSize = 8
dataShape = 256
numClasses = 2
box_per_cell = 2
mean_red, mean_blue, mean_green = 123, 117, 104
rgb_mean = nd.array([123, 117, 104])
rgb_std = nd.array([58.395, 57.12, 57.375])
maxEpoch = 1000
pretrained_params = '90.params'

ctx = mx.cpu()
#1--create data iter

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

net.collect_params().reset_ctx(ctx)
net.load_params(pretrained_params,ctx=ctx)
#check data shape in final layer
#X = nd.random_normal(0,1,shape=(batchSize, 3, dataShape, dataShape))
#Y = net.forward(X)

#3--define loss


#4--get trainer

net.collect_params().reset_ctx(ctx)
trainer = gluon.Trainer(net.collect_params(),"sgd",{"learning_rate":0.01,"wd":5e-4})



#5--start train
#5.1-parse net output
def convert_xy(XY):
    B,H,W,A,N = XY.shape
    dy = nd.tile( nd.arange(0,H,repeat=(W*A), ctx = XY.context).reshape((1,H,W,A,1)), (B,1,1,1,1) )
    dx = nd.tile( nd.arange(0,W,repeat=(A),ctx = XY.context).reshape((1,1,W,A,1)), (B,H,1,1,1) )
    x,y = XY.split(num_outputs=2,axis=-1)
    x = (x + dx) / W
    y = (y + dy) / H
    if 0:
        for b in range(B):
            for h in range(H):
                for w in range(W):
                    for a in range(A):
                        for n in range(1):
                            xx = dx[b,h,w,a,n].asnumpy()[0]
                            yy = dy[b,h,w,a,n].asnumpy()[0]
                            #pdb.set_trace()
                            print '(%.3f,%.3f)'%(xx,yy)
                    print ''
    return x,y
def convert_wh(WH):
    B,H,W,A,N = WH.shape
    w,h = WH.split(num_outputs=2,axis=-1)
    w = w ** 2
    h = h ** 2
    return w,h
def parse_net_output(Y,numClass, box_per_cell):
    pred = nd.transpose(Y,(0,2,3,1))
    pred = pred.reshape((0,0,0,box_per_cell,numClass + 5)) #add one dim for boxes
    predCls = nd.slice_axis(pred, begin = 0, end = numClass,axis=-1)
    predObject = nd.slice_axis(pred,begin=numClass,end=numClass+1,axis=-1)
    #predObject = nd.sigmoid(predObject)
    predXY = nd.slice_axis(pred, begin = numClass + 1, end = numClass + 3, axis=-1)
    predWH = nd.slice_axis(pred, begin = numClass + 3, end = numClass + 5, axis=-1)
    #predXY = nd.sigmoid(predXY)
    x,y = convert_xy(predXY)
    w,h = convert_wh(predWH)
    w = nd.clip(w,0,1)
    h = nd.clip(h,0,1)
    x0 = nd.clip(x, 0, 1)
    y0 = nd.clip(y,0,1)
    x1 = nd.clip(x0 + w,0,1)
    y1 = np.clip(y0 + h, 0,1)
    x = x0
    y = y0
    w = x1 - x0
    h = y1 - y0
    XYWH = nd.concat(x,y,w,h,dim=-1)
    return predCls, predObject, XYWH

def parse_groundtruth_for_target(labels, box_per_cell, xywh):
    B,H,W,A,_ = xywh.shape
    _,maxObjNum,_ = labels.shape
    boxMask = nd.zeros( (B,H,W,A,1), ctx = xywh.context )
    boxCls = nd.zeros((B,H,W,A,1), ctx = xywh.context)
    boxObject = nd.zeros((B,H,W,A,1),ctx = xywh.context)
    boxXYWH = nd.zeros((B,H,W,A,4), ctx = xywh.context)
    for b in range(B):
        label  = labels[b].asnumpy()
        validLabel = label[np.where(label[:,1] >-0.5)[0],:]
        np.random.shuffle(validLabel)
        for l in validLabel:
            cls,x,y,w,h,_ = l
            #find best box for this object
            indx,indy = int(x*W), int(y*H) #position
            pws, phs = xywh[b,indy, indx, :, -2], xywh[b,indy,indx,:,-1]
            ious = []
            pws = pws.asnumpy()
            phs = phs.asnumpy()
            for pw, ph in zip(pws,phs):
                intersect = np.minimum(pw,w) * np.maximum(ph,h)
                ious.append(  intersect / (pw * ph + w * h - intersect) )
            bestbox = int(np.argmax(ious))
            boxMask[b,indy,indx,bestbox,:] = 1.0
            boxCls[b,indy,indx,bestbox,:] = cls
            boxObject[b,indy,indx,bestbox,:] = 1.0 # ious[bestbox]
            tx = x * W - indx
            ty = y * H - indy
            tw,th = w,h
            boxXYWH[b,indy,indx,bestbox,:] = nd.array([tx,ty,tw,th])
    return boxMask, boxCls, boxObject,boxXYWH

#5.2 loss


#test
from mxnet import image
def process_image(fname):
    with open(fname, 'rb') as f:
        im = image.imdecode(f.read())
    # resize to data_shape
    data = image.imresize(im, dataShape, dataShape)
    # minus rgb mean, divide std
    data = (data.astype('float32') - rgb_mean) / rgb_std
    # convert to batch x channel x height xwidth
    return data.transpose((2,0,1)).expand_dims(axis=0), im

def predict(x):
    x = net(x)
    #pdb.set_trace()
    predCls, predObject, XYWH = parse_net_output(x, numClasses, box_per_cell)
    cid = nd.argmax(predCls,axis=-1,keepdims=True) #get cid as output
    #pdb.set_trace()
    output = nd.concat( cid, predObject, XYWH, dim=-1)
    output = output.reshape((0, -1, 6))
    #pdb.set_trace()
    #output[:,:,-2] = output[:,:,-2] ** 2 
    #output[:,:,-1] = output[:,:,-1] ** 2 #predict sqrt(w) sqrt(h) 
    output[:,:,-2] += output[:,:,-4]
    output[:,:,-1] += output[:,:,-3]
    #pdb.set_trace()
    zz = nd.contrib.box_nms(output) #change class id !!!
    return zz
    
x, im = process_image('img/test.jpg')
out = predict(x.as_in_context(ctx))
out.shape
out

import matplotlib as mpl
import matplotlib.pyplot as plt

#mpl.rcParams['figure.figsize'] = (6,6)
class_names = 'aeroplane,bicycle,bird,boat,bottle,bus,car,cat,chair,cow,diningtable,dog,horse,motorbike,person,pottedplant,sheep,sofa,train,tvmonitor'.split(',')
colors = ['blue', 'green', 'red', 'black', 'magenta']
def box_to_rect(box, color, linewidth=3):
    """convert an anchor box to a matplotlib rectangle"""
    box = box.asnumpy()
   
    return plt.Rectangle(
        (box[0], box[1]), box[2]-box[0], box[3]-box[1],
        fill=False, edgecolor=color, linewidth=linewidth)
def display(im, out, threshold=0.5):
    plt.imshow(im.asnumpy())
    for row in out:
        row = row.asnumpy()
        #pdb.set_trace()
        #print row
        class_id, score = int(row[0]), row[1]
        if class_id < 0 or score < threshold:
            continue
        print row
        color = colors[class_id%len(colors)]       
        #pdb.set_trace()
        box = row[-4:] * np.array([im.shape[0],im.shape[1]]*2)
        print box, class_id
        rect = box_to_rect(nd.array(box), color, 2)
        plt.gca().add_patch(rect)
        text = class_names[class_id]
     #   plt.gca().text(box[0], box[1],
     #                 '{:s} {:.2f}'.format(text, score),
     #                  bbox=dict(facecolor=color, alpha=0.5),
     #                  fontsize=10, color='white')
    plt.show()

display(im, out[0], threshold=0.90)    