import os,sys,pdb
import numpy as np
import mxnet as mx
from mxnet import gluon
from mxnet.gluon import nn
from mxnet import nd

class NetLayer(nn.Block):
    def __init__(self, channels, same_shape = True, **kwargs):
        super(NetLayer, self).__init__(**kwargs)
        self.same_shape = same_shape 
        strides = 1 if same_shape else 2
        self.conv1 = nn.Conv2D(channels, kernel_size = 3, strides = strides, padding=1)
        self.bn1 = nn.BatchNorm()
        return
    def forward(self,x):
        out = self.bn1( self.conv1(x) )
        return nd.tanh(out)

class ConvNet(nn.Block):
    def __init__(self, num_class, verbose = False, **kwargs):
        super(ConvNet,self).__init__(**kwargs)
        self.verbose = verbose
        with self.name_scope():
            layers = []
            for stage in range(20): #20 stages conv+bn+relu 
                if stage < 8:
                    layers.append( NetLayer(16, same_shape=True) )
                elif stage == 8:
                    layers.append( NetLayer(32, same_shape=False) )
                elif stage < 12:
                    layers.append( NetLayer(32, same_shape = True) )
                elif stage == 12:
                    layers.append( NetLayer(64, same_shape = False))
                elif stage < 14:
                    layers.append( NetLayer(64, same_shape = True) )
                elif stage == 14:
                    layers.append( NetLayer(128, same_shape = False) )
                elif stage < 18:
                    layers.append( NetLayer(128, same_shape = True) )
                elif stage == 18:
                    layers.append( NetLayer(256, same_shape = False) )
                else:
                    layers.append( NetLayer(256, same_shape = True))
            layers.append( nn.GlobalAvgPool2D() ) #21 average pool
            layers.append( nn.Dense(num_class) )  #22 IP
                
            self.net = nn.Sequential()
            for layer in layers:
                self.net.add(layer)
            
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
    net = ConvNet(num_class,verbose = True)
    net.initialize(ctx=[mx.cpu()])
    x = mx.nd.random.uniform(0,1,shape=(3,3,64,64))
    y = net(x)
    print y.shape
    
    net = ConvNet(num_class,verbose = False)
    net.initialize(ctx = ctx)
    return net
    
    