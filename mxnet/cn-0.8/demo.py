import mxnet as mx
from mxnet.gluon import nn
from mxnet import gluon
import sys
import utils
import pdb,os,sys
from importlib import import_module
import logging

trainBatchSize = 50
testBatchSize = 50
dataShape = (3,48,48)
classNum = 11
pretrained = 'cp/epoch-000001.params'

lr_base = 0.01

 
def get_net():      
    mod = import_module('symbol.convnet')
    net = mod.get_symbol(classNum,utils.try_gpu())
    return net 

def get_train_test():
    aug = mx.image.CreateAugmenter(data_shape = dataShape, resize = dataShape[1], mean=True, std=True)

    trainIter = mx.image.ImageIter(batch_size=trainBatchSize, data_shape=dataShape, path_imgrec='train.rec',\
                path_imgidx = 'train.idx',
                aug_list=aug)
    testIter = mx.image.ImageIter(batch_size=testBatchSize, data_shape=dataShape, path_imgrec='test.rec',\
                path_imgidx = 'test.idx',
                aug_list=aug)
    return (trainIter, testIter)
    
def get_trainer(net):      
    lr_sch = mx.lr_scheduler.FactorScheduler(step=1000,factor=0.1)
    loss = gluon.loss.SoftmaxCrossEntropyLoss()
    trainer = gluon.Trainer(net.collect_params(),"sgd",{'learning_rate':lr_base,'lr_scheduler':lr_sch})
    return (trainer,loss)
    
def main():
    
    net = get_net()
    net_str = '%s'%net
    logging.info('ok')
    logging.info(net_str)
    if pretrained is not None:
        net.load_params(pretrained,ctx=utils.try_gpu())
    trainIter, testIter = get_train_test()
    trainer,loss = get_trainer(net)
    utils.train(trainIter, testIter, net, loss, trainer, utils.try_gpu(), 1000,print_batches=100, cpdir='cp')
    
if __name__=="__main__":
    logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p',filename="train.log", level=logging.INFO)
    main()



