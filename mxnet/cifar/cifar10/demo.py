import mxnet as mx
from mxnet.gluon import nn
from mxnet import gluon
import sys
import utils
import pdb,os,sys
from importlib import import_module
import logging
import numpy as np

trainBatchSize = 100
testBatchSize = 50
dataShape = (3,32,32)
classNum = 10
pretrained = None
checkpoints = 'checkpoints/'
inputroot = "c:/dataset/cifar/split/"

lr_base = 0.01
weight_decay = 0.0005

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
    
def train_transform(X,Y):
    return test_transform(X,Y)
    
 
def get_net():      
    mod = import_module('symbol.resnet18')
    net = mod.get_symbol(classNum,utils.try_gpu())
    return net 

def get_train_test(): #mxnet 1.0.0
    train_ds = mx.gluon.data.vision.ImageFolderDataset( os.path.join(inputroot, 'train') , flag=1, transform = train_transform)
    test_ds = mx.gluon.data.vision.ImageFolderDataset( os.path.join(inputroot, 'test'), flag=1, transform = test_transform)
    for label,labelname in enumerate( train_ds.synsets ):
        logging.info('%d %s'%(label, labelname))
    loader = mx.gluon.data.DataLoader
    train_data = loader( train_ds, \
                trainBatchSize,shuffle=True, last_batch='keep')
    test_data =loader( test_ds, \
                testBatchSize, shuffle=True, last_batch='keep')
    return (train_data, test_data)
    
def get_trainer(net):      
    loss = gluon.loss.SoftmaxCrossEntropyLoss()
    trainer = gluon.Trainer(net.collect_params(),"sgd",{'learning_rate':lr_base, 'momentum':0.9, 'wd':weight_decay})
    return (trainer,loss)
    
def main():
    
    net = get_net()
    net_str = '%s'%net
    #logging.info('ok')
    logging.info(net_str)
    if pretrained is not None:
        net.load_params(pretrained,ctx=utils.try_gpu())
    train_data, test_data = get_train_test()
    trainer,loss = get_trainer(net)
    utils.train(train_data, test_data, trainBatchSize,\
        net, loss, trainer, utils.try_gpu(), 1000,\
        500,0.1,print_batches=100, chk_pts_dir=checkpoints)
    
if __name__=="__main__":
    logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p',filename="train.log", level=logging.INFO)
    main()



