import mxnet as mx
from mxnet.gluon import nn
from mxnet import gluon
import sys
import utils
import pdb,os,sys
from importlib import import_module
import logging
import numpy as np

import pandas as pd

pretrained = '2.params'
testBatchSize = 50
classNum = 10
inputroot = "c:/dataset/cifar/split/"
dataShape = (3,32,32)

mean = np.zeros(dataShape)
mean[0,:,:] = 0.4914
mean[1,:,:] = 0.4822
mean[2,:,:] = 0.4465
std = np.zeros(dataShape)
std[0,:,:] = 0.2023
std[1,:,:] = 0.1994
std[2,:,:] = 0.2010

def test_transform(X,Y):
    out = X.astype(np.float32)/255.0
    out = np.transpose(out,(2,0,1))
    #pdb.set_trace()
    #return (mx.image.color_normalize(out,np.asarray([0.4914, 0.4822, 0.4465]), np.asarray([0.2023, 0.1994, 0.2010])),Y)
    return (mx.image.color_normalize(out.asnumpy(),mean,std),Y)
    
ctx = utils.try_gpu()
mod = import_module('symbol.resnet18')
net = mod.get_symbol(classNum,ctx)
net.load_params(pretrained,ctx=ctx)

test_ds = mx.gluon.data.vision.ImageFolderDataset( os.path.join(inputroot, 'test'), flag=1, transform = test_transform)
loader = mx.gluon.data.DataLoader
test_data = loader( test_ds, testBatchSize, shuffle=False, last_batch='keep')


preds = []
for data, label in test_data:
    output = net(data.as_in_context(ctx))
    preds.extend(output.argmax(axis=1).astype(int).asnumpy())

hit = 0
for pred, groundtruth in zip(preds, test_ds.items):
    y0 = test_ds.synsets[groundtruth[1]]
    y1 = test_ds.synsets[pred]
    #print groundtruth[0], y0, y1
    if y0 == y1:
        hit += 1
print hit, len(preds), np.float64(hit) / len(preds)
        
    
    