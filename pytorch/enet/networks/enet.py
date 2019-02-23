from __future__ import print_function
import torch
import torch.nn as nn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import os,copy

print("pytorch vision: ",torch.__version__)
print("Torchvision Version: ",torchvision.__version__)



class ENET_CONV(nn.Module):
    def __init__(self,in_channels, out_channels,kernel_size,conv_type="normal", dilation=1, strides=(1,1)):
        super(ENET_CONV,self).__init__()
        self.conv_type = conv_type
        if conv_type == "dilated":
            self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                  stride=strides,padding=kernel_size*dilation//2,bias=False)
        else:
            self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,kernel_size=kernel_size,
                                  stride=strides,padding=kernel_size//2,bias=False)
        self.upsampling = nn.UpsamplingBilinear2d(scale_factor=2)
        self.prelu = nn.PReLU()
        self.bn = nn.BatchNorm2d(num_features=out_channels)
    def forward(self,x):
        if self.conv_type == "upsampling":
            x = self.upsampling(x)
        out = self.prelu( self.bn(self.conv(x)) )
        return out


class ENET_INIT(nn.Module):
    def __init__(self):
        super(ENET_INIT,self).__init__()
        self.maxpool = nn.MaxPool2d(kernel_size=(2,2),stride=2,padding=0)
        self.conv = ENET_CONV(in_channels=3, out_channels=13,kernel_size=3,strides=2)
        return
    def forward(self, x):
        x1 = self.maxpool(x)
        x2 = self.conv(x)
        return torch.cat((x1,x2),dim=1)

class ENET_BOTTLENECK(nn.Module):
    def __init__(self,in_channels, out_channels,downsampling=False,conv_type = "normal",kernel_size=(3,3)):
        super(ENET_BOTTLENECK,self).__init__()
        self.downsampling = downsampling
        if downsampling:
            self.b1 = nn.Sequential(
                ENET_CONV(in_channels=in_channels, out_channels=out_channels,kernel_size=3,strides=2),
                ENET_CONV(conv_type=conv_type,in_channels=out_channels,out_channels=out_channels,kernel_size=kernel_size,strides=1),
                ENET_CONV(in_channels=out_channels,out_channels=out_channels,kernel_size=1,strides=1)
            )
            self.regular = nn.Dropout2d()
            self.b2 = nn.Sequential(
                nn.MaxPool2d(kernel_size=2,stride=2,padding=0),
                nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=1,stride=1,padding=0,bias=False)
            )
            self.conv = nn.Conv2d(in_channels=out_channels,out_channels=out_channels,kernel_size=1,stride=1,padding=0,bias=False)
            self.prelu = nn.PReLU()
        else:
            self.b1 = nn.Sequential(
                ENET_CONV(in_channels=in_channels,out_channels=out_channels,kernel_size=1,strides=1),
                ENET_CONV(conv_type=conv_type,in_channels=out_channels,out_channels=out_channels,
                          kernel_size=kernel_size,strides=1),
                ENET_CONV(in_channels=out_channels,out_channels=out_channels,kernel_size=1,strides=1)
            )


        return
    def forward(self,x):
        x1 = x
        for layer in self.b1:
            x1 = layer(x1)
        if self.downsampling:
            x1 = self.regular(x1)
            x2 = self.b2(x)
            return self.prelu( self.conv(x1 + x2) )
        return x1


class ENET(nn.Module):
    def __init__(self, num_class):
        super(ENET,self).__init__()
        self.stages = nn.Sequential(
            ENET_INIT(),

            ENET_BOTTLENECK(16,64,downsampling=True,kernel_size=3),
            ENET_BOTTLENECK(64,64,kernel_size=3),
            ENET_BOTTLENECK(64,64,kernel_size=3),
            ENET_BOTTLENECK(64,64,kernel_size=3),
            ENET_BOTTLENECK(64,64,kernel_size=3),


            ENET_BOTTLENECK(64,128,downsampling=True,kernel_size=3),
            ENET_BOTTLENECK(128,128,kernel_size=3),
            ENET_BOTTLENECK(128,128,kernel_size=3,conv_type="dilated"), #2
            ENET_BOTTLENECK(128,128,kernel_size=5,conv_type="asymmetric"),
            ENET_BOTTLENECK(128,128,kernel_size=3,conv_type="dilated"), #4
            ENET_BOTTLENECK(128,128,kernel_size=3),
            ENET_BOTTLENECK(128,128,kernel_size=3,conv_type="dilated"), #8
            ENET_BOTTLENECK(128,128,kernel_size=5,conv_type="asymmetric"),
            ENET_BOTTLENECK(128,128,kernel_size=3,conv_type="dilated"), #16

            ENET_BOTTLENECK(128,128,kernel_size=3),
            ENET_BOTTLENECK(128,128,kernel_size=3,conv_type="dilated"), #2
            ENET_BOTTLENECK(128,128,kernel_size=5,conv_type="asymmetric"),
            ENET_BOTTLENECK(128,128,kernel_size=3,conv_type="dilated"), #4
            ENET_BOTTLENECK(128,128,kernel_size=3),
            ENET_BOTTLENECK(128,128,kernel_size=3,conv_type="dilated"), #8
            ENET_BOTTLENECK(128,128,kernel_size=5,conv_type="asymmetric"),
            ENET_BOTTLENECK(128,128,kernel_size=3,conv_type="dilated"), #16


            ENET_BOTTLENECK(128,64,kernel_size=3,conv_type="upsampling"),
            ENET_BOTTLENECK(64,64,kernel_size=3),
            ENET_BOTTLENECK(64,64,kernel_size=3),

            ENET_BOTTLENECK(64,16,kernel_size=3,conv_type="upsampling"),
            ENET_BOTTLENECK(16,16,kernel_size=3),
        )
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)
        self.lastConv = nn.Conv2d(16,num_class,kernel_size=3,stride=1,padding=1,bias=False)
        return

    def forward(self, x):
        out = self.stages(x)
        out = self.upsample(out)
        out = self.lastConv(out)
        return out

def get_net(num_class):
    return ENET(num_class)


if 0:
    x = torch.zeros((2,3,512//2,512//2), dtype=torch.float32)  # minibatch size 64, feature dimension 50
    model = get_net(21)
    scores = model(x)
    print(scores.size())  # you should see [64, 10]

#https://github.com/szagoruyko/pytorchviz
#pip install git+https://github.com/szagoruyko/pytorchviz
if 0:
    from graphviz import Digraph
    from torchviz import make_dot
    from torch.autograd import Variable

    net = get_net(21)
    net.cuda()
    x = Variable(torch.rand(1, 3, 256, 256)).cuda()
    h_x = net(x).cpu()
    dot = make_dot(h_x, params=dict(net.named_parameters()))
    dot.view("enet")


