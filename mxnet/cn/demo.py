import mxnet as mx
from mxnet.gluon import nn
from mxnet import gluon
import sys
import utils
import pdb,os,sys

trainBatchSize = 50
testBatchSize = 50
dataShape = (3,200,200)
imgroot="c:/dataset/colorname3/256x256/"

net = nn.Sequential()
with net.name_scope():
    #L1
    net.add( nn.Conv2D(channels=96, kernel_size=11, strides = 4 ) )
    net.add( nn.BatchNorm(axis=1) )
    net.add( nn.Activation(activation = "relu" ) )
    net.add( nn.MaxPool2D(pool_size=3, strides=2) )
    #L2
    net.add( nn.Conv2D(channels=256, kernel_size=5, strides = 1, padding = 2) )
    net.add( nn.BatchNorm(axis=1) )
    net.add( nn.Activation(activation="relu"))
    net.add( nn.MaxPool2D(pool_size=3, strides=2))
    #L3
    net.add( nn.Conv2D(channels=384, kernel_size=3, strides = 1, padding = 1))
    net.add( nn.BatchNorm(axis=1) )
    net.add( nn.Activation(activation="relu"))
    #L4
    net.add( nn.Conv2D(channels=384, kernel_size=3, strides=1, padding=1))
    net.add( nn.BatchNorm(axis=1) )
    net.add( nn.Activation(activation="relu"))
    #L5
    net.add( nn.Conv2D(channels=256, kernel_size=3, strides=1, padding=1) )
    net.add( nn.BatchNorm(axis=1) )
    net.add( nn.Activation(activation="relu"))
    net.add( nn.MaxPool2D(pool_size=3, strides=2) )
    #L6
    net.add( nn.Dense(4096, activation="relu"))
    net.add( nn.Dropout(0.5))
    #L7
    net.add( nn.Dense(4096, activation="relu"))
    net.add( nn.Dropout(0.5))
    #L8
    net.add( nn.Dense(11) )

#mx.viz.plot_network(net).view()
net.initialize( ctx=utils.try_gpu() )
aug = mx.image.CreateAugmenter(data_shape = dataShape, resize = dataShape[1], mean=True, std=True)

trainIter = mx.image.ImageIter(batch_size=trainBatchSize, data_shape=dataShape, path_imgrec='train.rec',\
            path_imgidx = 'train.idx',
            aug_list=aug)
testIter = mx.image.ImageIter(batch_size=testBatchSize, data_shape=dataShape, path_imgrec='test.rec',\
            path_imgidx = 'test.idx',
            aug_list=aug)
lr_sch = mx.lr_scheduler.FactorScheduler(step=100,factor=0.8)
loss = gluon.loss.SoftmaxCrossEntropyLoss()
trainer = gluon.Trainer(net.collect_params(),"sgd",\
{'learning_rate':0.01,'lr_scheduler':lr_sch})
utils.train(trainIter, testIter, net, loss, trainer, utils.try_gpu(), 1000,print_batches=100)



