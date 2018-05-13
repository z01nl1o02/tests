import mxnet as mx
from mxnet.gluon import nn
from mxnet import gluon
from mxnet import ndarray as nd
from mxnet import io
import sys,random
import utils
import pdb,os,sys
from importlib import import_module
import logging
import cv2
import numpy as np

dataShape = (3,24,24)
outputNum = 1*2
path_root="c:/dataset/landmark/train/for-mxnet/RE/"
roiSize = 25
landmarkIdx = 1

trainBatchSize = 50
testBatchSize = 20
pretrained = None
lr_base = 0.5

 
"""
mod = import_module('symbol.facenet')
net = mod.get_symbol(outputNum,utils.try_gpu(),verbose=True)
x = mx.nd.random.uniform(0,1,shape=(3,1,39,39))
y = net(x)
print y.shape
pdb.set_trace()
"""


class MyImgIter(mx.image.ImageIter):
  def __init__(self, landmarkIdx, shuffleRatio, cropSize, stdSize, batch_size, data_shape, label_width=1,
                 path_imgrec=None, path_imglist=None, path_root=None, path_imgidx=None,
                 shuffle=False, part_index=0, num_parts=1, aug_list=None, imglist=None,
                 data_name='data', label_name='softmax_label', **kwargs):
    super(MyImgIter,self).__init__(batch_size, data_shape, label_width, path_imgrec, path_imglist, path_root,\
                                     path_imgidx, shuffle, part_index, num_parts, aug_list, imglist,\
                                     data_name, label_name, **kwargs)
    self.shuffleRatio = shuffleRatio
    self.landmarkIdx = landmarkIdx
    self.stdSize = stdSize
    self.cropSize = cropSize
  def aug_position(self,data,label):
    H,W,C = data.shape
    dx = np.int64(W * self.shuffleRatio)
    dy = np.int64(H * self.shuffleRatio)
    x,y = label[self.landmarkIdx*2] * W, label[self.landmarkIdx*2+1]*H
    dx = random.randint(-dx,dx)
    dy = random.randint(-dy,dy)
    x0 = x + dx - self.cropSize / 2
    y0 = y + dy - self.cropSize / 2
    if x0 < 0:
        x0 = 0
    if y0 < 0:
        y0 = 0
    x1 = x0 + self.cropSize
    y1 = y0 + self.cropSize
    if x1 >= W:
        x1 = W
    if y1 >= H:
        y1 = H
    if x1 - x0 < self.cropSize / 3 or y1 - y0 < self.cropSize / 3:
        return (None,None)
    x0,x1,y0,y1 = [np.int64(k) for k in [x0,x1,y0,y1]]
    res = cv2.resize( data[y0:y1,x0:x1,:], (self.stdSize,self.stdSize) )
    return (res, np.asarray([ -dx*1.0/W, -dy*1.0/H]) ) #switch to offset  

  def next(self):
    """Returns the next batch of data."""
    batch_size = self.batch_size
    c, h, w = self.data_shape
    batch_data = nd.empty((batch_size, c, h, w))
    batch_label = nd.empty(self.provide_label[0][1])
    i = 0
    try:
        while i < batch_size:
            #pdb.set_trace()
            label, s = self.next_sample()        
            data= self.imdecode(s)
            filepath = os.path.join(self.path_root,self.imglist[self.seq[self.cur - 1]][1])
            data = cv2.imread(filepath, 1)
            data,label = self.aug_position(data,label.asnumpy())           
            data = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
            data = mx.nd.array(data).as_in_context(mx.cpu())
            label = nd.array(label).as_in_context(mx.cpu())
            try:
                self.check_valid_image(data)
            except RuntimeError as e:
                logging.debug('Invalid image, skipping:  %s', str(e))
                continue
            data = self.augmentation_transform(data)
            assert i < batch_size, 'Batch size must be multiples of augmenter output length'
            batch_data[i] = self.postprocess_data(data)
            batch_label[i] = label
            i += 1
    except StopIteration:
        if not i:
            raise StopIteration

    return io.DataBatch([batch_data], [batch_label], batch_size - i)

def get_net():      
    mod = import_module('symbol.facenetS2')
    net = mod.get_symbol(outputNum,utils.try_gpu(),verbose=False)
    return net 

    

def get_train_test():
    trainAug = mx.image.CreateAugmenter(data_shape = dataShape, resize = dataShape[1], mean=True, std=True,\
            pca_noise=0.05,contrast=0.125, brightness=0.125,saturation=0.125)
    testAug = mx.image.CreateAugmenter(data_shape = dataShape, resize = dataShape[1], mean=True, std=True)

    trainIter = MyImgIter(landmarkIdx,0.05, roiSize, dataShape[1], batch_size=trainBatchSize, data_shape=dataShape, 
                label_width = outputNum,
                path_imglist=os.path.join(path_root,'train/landmarks.lst'),
                path_root= os.path.join( path_root, 'train'),
                shuffle = True,
                aug_list=trainAug)
    testIter = MyImgIter(landmarkIdx,0.0, roiSize, dataShape[1],batch_size=testBatchSize, data_shape=dataShape, 
                label_width = outputNum,
                path_imglist=os.path.join(path_root,'test/landmarks.lst'),
                path_root=os.path.join( path_root, 'test'),
                aug_list=testAug)
    return (trainIter, testIter)
    
def get_trainer(net):      
    trainer = gluon.Trainer(net.collect_params(),"sgd",{'learning_rate':lr_base,'wd':0.0000})
    lossfunc = gluon.loss.L2Loss()
    return trainer,lossfunc
    
def main():
    net = get_net()
    net_str = '%s'%net
    logging.info(net_str)
    if pretrained is not None and pretrained != "":
        net.load_params(pretrained,ctx=utils.try_gpu())
        logging.info("load model:%s"%pretrained)
    trainIter, testIter = get_train_test()
    trainer,lossfunc = get_trainer(net)
    lr_steps = [6000,12000,18000,24000]
    utils.train(trainIter, testIter, net,lossfunc, trainer, utils.try_gpu(), 100000,lr_steps,print_batches=10, cpdir='models')
    
if __name__=="__main__":
    logging.basicConfig(format='%(asctime)s %(message)s', filemode='w',datefmt='%m/%d/%Y %I:%M:%S %p',filename="train.log", level=logging.INFO)

    main()



