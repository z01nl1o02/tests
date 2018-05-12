import mxnet as mx
from mxnet.gluon import nn
from mxnet import gluon
import sys
import utils
import pdb,os,sys
from importlib import import_module
import logging

dataShape = (3,56,64)
outputNum = 3*2
path_root="c:/dataset/landmark/train/for-mxnet/NM1/"


trainBatchSize = 60
testBatchSize = 20
pretrained = None
lr_base = 0.5

 
"""
mod = import_module('symbol.facenet')
net = mod.get_symbol(outputNum,utils.try_gpu(),verbose=True)
x = mx.nd.random.uniform(0,1,shape=(3,3,56,64))
y = net(x)
print y.shape
pdb.set_trace()
"""

def get_net():      
    mod = import_module('symbol.facenet')
    net = mod.get_symbol(outputNum,utils.try_gpu(),verbose=False)
    return net 

    

def get_train_test():
    aug = mx.image.CreateAugmenter(data_shape = dataShape, resize = dataShape[1], mean=True, std=True)

    trainIter = mx.image.ImageIter(batch_size=trainBatchSize, data_shape=dataShape, 
                label_width = outputNum,
                path_imglist=os.path.join(path_root,'train/landmarks.lst'),
                path_root= os.path.join( path_root, 'train'),
                shuffle = True,
                aug_list=aug)
    testIter = mx.image.ImageIter(batch_size=testBatchSize, data_shape=dataShape, 
                label_width = outputNum,
                path_imglist=os.path.join(path_root,'test/landmarks.lst'),
                path_root=os.path.join( path_root, 'test'),
                aug_list=aug)
    return (trainIter, testIter)
    
def get_trainer(net):      
    trainer = gluon.Trainer(net.collect_params(),"sgd",{'learning_rate':lr_base,'wd':0.0005})
    lossfunc = gluon.loss.L2Loss()
    return trainer,lossfunc
    
def main():
    net = get_net()
    net_str = '%s'%net
    logging.info(net_str)
    if pretrained is not None:
        net.load_params(pretrained,ctx=utils.try_gpu())
    trainIter, testIter = get_train_test()
    trainer,lossfunc = get_trainer(net)
    lr_steps = [400, 800, 1200,2000,3500]
    utils.train(trainIter, testIter, net,lossfunc, trainer, utils.try_gpu(), 10000,lr_steps,print_batches=10, cpdir='models')
    
if __name__=="__main__":
    logging.basicConfig(format='%(asctime)s %(message)s', filemode='w',datefmt='%m/%d/%Y %I:%M:%S %p',filename="train.log", level=logging.INFO)

    main()



