import os,sys,pdb
import mxnet as mx
import numpy as np

class FACENET_LAYER(mx.gluon.nn.Block):
    def __init__(self,channels,kernelSize,**kwargs):
        super(FACENET_LAYER,self).__init__(**kwargs)
        self.conv = mx.gluon.nn.Conv2D(channels,kernel_size = kernelSize, strides = 1, padding=0)
        #self.bn = mx.gluon.nn.BatchNorm()
        return
    def forward(self,x):
        #return self.conv(x)
        #out = self.bn(self.conv(x))
        return mx.nd.abs( mx.nd.tanh(self.conv(x)) )
        
class FACENET(mx.gluon.nn.Block):
    def __init__(self, outputNum, verbose=False, **kwargs):
        super(FACENET,self).__init__(**kwargs)
        self.verbose = verbose
        with self.name_scope():
            layers = []
            layers.append( FACENET_LAYER(20,4) )
            layers.append( mx.gluon.nn.MaxPool2D( pool_size=(2,2), strides=2 ) )
            layers.append( FACENET_LAYER(40,3) )
            layers.append( mx.gluon.nn.MaxPool2D( pool_size=(2,2), strides=2 ) )
            layers.append( FACENET_LAYER(60,3) )
            layers.append( mx.gluon.nn.MaxPool2D( pool_size=(2,2), strides=2 ) )
            layers.append( FACENET_LAYER(80,2) )
            layers.append( mx.gluon.nn.Dense(120, activation ='tanh') )
            layers.append( mx.gluon.nn.Dense(outputNum, activation ='tanh') )
        self.net = mx.gluon.nn.Sequential()
        for layer in layers:
            self.net.add( layer )
        return
    def forward(self,x):
        out = x
        for i, layer in enumerate(self.net):
            out = layer(out)
            if self.verbose:
                print('block %d output %s'%(i+1,out.shape))
        return out
            
            
def get_symbol(outputNum,ctx,verbose = False,**kwargs):
    if verbose:
        net = FACENET(outputNum,verbose = True)
        net.initialize(ctx=[mx.cpu()])
    else:
        net = FACENET(outputNum)
        net.initialize(ctx = ctx)
    return net