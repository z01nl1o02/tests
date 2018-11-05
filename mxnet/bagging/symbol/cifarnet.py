import numpy as np
import mxnet as mx
import cv2
from mxnet import gluon
from mxnet.gluon import Block

class CIFARNET_BLOCK(Block):
    def __init__(self,channels,pool_type,ctx, **kwargs):
        super(CIFARNET_BLOCK,self).__init__(**kwargs)
        with self.name_scope():
            self.layers = gluon.nn.Sequential()
            self.layers.add(
                gluon.nn.Conv2D(channels=channels,kernel_size=5,strides=1,padding=2),
                gluon.nn.Activation(activation="relu"),
            )
            if pool_type == "ave":
                self.layers.add(
                    gluon.nn.AvgPool2D(pool_size=3, strides=2)
                )
            if pool_type == "max":
                self.layers.add(
                    gluon.nn.MaxPool2D(pool_size=3, strides=2)
                )

            for layer in self.layers:
                if isinstance(layer,gluon.nn.Conv2D):
                    layer.initialize(init = mx.initializer.Xavier(),ctx=ctx)
                else:
                    layer.initialize(ctx=ctx)
            return
    def forward(self, *args):
        out = args[0]
        for layer in self.layers:
            out = layer(out)
        return out


class CIFARNET_BLOCK_A(Block):
    def __init__(self, channels, pool_type, ctx, **kwargs):
        super(CIFARNET_BLOCK_A, self).__init__(**kwargs)
        with self.name_scope():
            self.layers = gluon.nn.Sequential()
            self.layers.add(
                gluon.nn.Conv2D(channels=channels, kernel_size=5, strides=1, padding=2,bias_initializer=mx.init.Constant(0)),
                gluon.nn.MaxPool2D(pool_size=3,strides=2),
                gluon.nn.Activation(activation="relu"),

            )
            for layer in self.layers:
                if isinstance(layer, gluon.nn.Conv2D):
                    layer.initialize(init=mx.initializer.Xavier(), ctx=ctx)
                else:
                    layer.initialize(ctx=ctx)
            return

    def forward(self, *args):
        out = args[0]
        for layer in self.layers:
            out = layer(out)
        return out


class CIFARNET_QUICK(Block):
    def __init__(self, class_num, ctx, **kwargs):
        super(CIFARNET_QUICK,self).__init__(**kwargs)
        with self.name_scope():
            self.layers = gluon.nn.Sequential()
            self.layers.add(
                CIFARNET_BLOCK_A(32,"max",ctx),
                CIFARNET_BLOCK(32,"max",ctx),
                CIFARNET_BLOCK(64,"max",ctx),
            )
            self.layers.add(
                gluon.nn.Dense(64),
                gluon.nn.Dropout(0.5),
                gluon.nn.Dense(class_num)
            )
            self.layers[-3].initialize(init = mx.initializer.Xavier(),ctx=ctx)
            self.layers[-2].initialize(ctx=ctx)
            self.layers[-1].initialize(init = mx.initializer.Xavier(),ctx=ctx)
        return
    def forward(self, *args):
        out = args[0]
        for layer in self.layers:
            out = layer(out)
        return out

if 0:
    import random
    ctx = mx.gpu()
    net = CIFARNET_QUICK(class_num=10,ctx=ctx)
    print net
    data = mx.nd.array( np.zeros((2,3,32,32)), ctx=ctx)
    output = net(data)
    print output.shape
