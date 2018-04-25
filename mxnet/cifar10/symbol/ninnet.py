import os,sys,pdb
import numpy as np
import mxnet as mx
from mxnet import gluon
from mxnet.gluon import nn


class NIN(nn.Block):
    def __init__(self, channels, kernel_size, padding, strides=1, max_pooling=True, **kwargs):
        super(NIN, self).__init__(**kwargs)
        self.conv1 = nn.Conv2D(channels=channels,kernel_size=kernel_size, strides=strides, padding=padding,
            activation ='relu')
        self.conv2 = nn.Conv2D(channels=channels, kernel_size=1, strides=1, padding=0, activation='relu')
        self.conv3 = nn.Conv2D(channels=channels, kernel_size=1, strides=1, padding=0, activation='relu')
        self.pool4 = None
        if max_pooling:
            self.pool4 = nn.MaxPool2D(pool_size=3,strides=2)
        return
    def forward(self,x):
        out = self.conv3( self.conv2( self.conv1(x) ) )
        if self.pool4 is not None:
            out = self.pool4(out)
        return out
class NINNet(nn.Block):
    def __init__(self, num_classes, **kwargs):
        super(NINNet, self).__init__(**kwargs)

        with self.name_scope():
            b1 = NIN(96,11,0,strides=4)
            b2 = NIN(256,5,2)
            b3 = NIN(384,3,1)
            b4 = nn.Dropout(0.5)
            #replace Dense()
            b5 = NIN(num_classes, 3, 1, max_pooling=False)
            b6 = nn.GlobalAvgPool2D()
            b7 = nn.Flatten()
                
        self.net = nn.Sequential()
        self.net.add(b1,b2,b3,b4,b5,b6,b7)
        return
    def forward(self, x):
        out = x
        for i, blk in enumerate(self.net):
            out = blk(out)
        return out 
        
def get_symbol(num_class,ctx,**kwargs):
    net = NINNet(num_class)
    net.initialize(ctx = ctx)
    return net
            
            
        