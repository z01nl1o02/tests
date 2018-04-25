import os,sys,pdb
import numpy as np
import mxnet as mx
from mxnet import gluon
from mxnet.gluon import nn
from mxnet import nd

class Residual(nn.Block):
    def __init__(self, channels, same_shape = True, **kwargs):
        super(Residual, self).__init__(**kwargs)
        self.same_shape = same_shape 
        strides = 1 if same_shape else 2
        self.conv1 = nn.Conv2D(channels, kernel_size = 3, strides = strides, padding=1)
        self.bn1 = nn.BatchNorm()
        self.conv2 = nn.Conv2D(channels, kernel_size = 3, strides = 1, padding=1)
        self.bn2 = nn.BatchNorm()
        if not same_shape:
            self.conv3 = nn.Conv2D(channels, kernel_size = 1, strides = strides)
        return
    def forward(self,x):
        out = self.bn1( self.conv1(x) )
        out = nd.relu(out)
        out = self.bn2( self.conv2(out) )
        if not self.same_shape:
            x = self.conv3(x)
        return nd.relu(out + x)

class ResNet(nn.Block):
    def __init__(self, num_class, verbose = False, **kwargs):
        super(ResNet,self).__init__(**kwargs)
        self.verbose = verbose
        with self.name_scope():
            #block 1
            b1 = nn.Conv2D(64, kernel_size=7, strides=2, padding=0)
            #block 2
            b2 = nn.Sequential()
            b2.add(
                nn.MaxPool2D(pool_size = 3, strides = 2),
                Residual(64),
                Residual(64)
                )
            #block 3
            b3 = nn.Sequential()
            b3.add( 
                Residual(128, same_shape = False),
                Residual(128)
                )
            #block 4
            b4 = nn.Sequential()
            b4.add(
                Residual(256, same_shape = False),
                Residual(256)
               )
            #block 5
            b5 = nn.Sequential()
            b5.add(
                Residual(512, same_shape = False),
                Residual(512)
                )
            #block 6
            b6 = nn.Sequential()
            b6.add(
                nn.AvgPool2D(pool_size = 3),
                nn.Dense(num_class)
                )
            self.net = nn.Sequential()
            self.net.add(b1,b2,b3,b4,b5,b6)
            
    def forward(self,x):
        out = x
        for i, b in enumerate(self.net):
            out = b(out)
            if self.verbose:
                print('block %d output: %s'%(i+1, out.shape))
        return out
        
def get_symbol(num_class, ctx, **kwargs):

    #blk = Residual(3,same_shape=False)
    #blk.initialize()
    #x = nd.random.uniform(shape=(4, 3, 6, 6))
    #print(blk(x).shape)

    net = ResNet(num_class)
    net.initialize(ctx = ctx)
    return net
    
    