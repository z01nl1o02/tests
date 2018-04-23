import mxnet as mx
from mxnet.gluon import nn
from mxnet import gluon
import sys
import utils
import pdb,os,sys
from importlib import import_module

trainBatchSize = 10
testBatchSize = 10
dataShape = (3,200,200)
classNum = 11

def get_net():      
    mod = import_module('symbol.resnet')
    net = mod.get_symbol(classNum,utils.try_gpu())
    print(net)
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
    lr_sch = mx.lr_scheduler.FactorScheduler(step=100,factor=0.9)
    loss = gluon.loss.SoftmaxCrossEntropyLoss()
    trainer = gluon.Trainer(net.collect_params(),"sgd",{'learning_rate':0.001,'lr_scheduler':lr_sch})
    return (trainer,loss)
    
def main():
    net = get_net()
    trainIter, testIter = get_train_test()
    trainer,loss = get_trainer(net)
    utils.train(trainIter, testIter, net, loss, trainer, utils.try_gpu(), 1000,print_batches=100)
    
if __name__=="__main__":
    main()



