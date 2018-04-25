import os,sys,pdb
import numpy as np
import mxnet as mx
from mxnet import gluon
from mxnet.gluon import nn
def get_symbol(num_class, ctx, **kwargs):
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
        net.add( nn.Dense(num_class) )
    net.initialize(ctx = ctx)
    return net
    
    