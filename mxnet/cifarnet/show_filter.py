import os,sys,cv2
import mxnet as mx
from mxnet import nd
import numpy as np
from config import config
from symbol.cifarnet import CIFARNET_QUICK,CIFARNET_BLOCK
from tqdm import tqdm

params_file = 'models_mnist/cifarnet_epoch_{:0>5d}.params'.format(198)
vis_block_size = 10 #larger than filter size

ctx = mx.cpu()
crop_hw = (config.height, config.width)
net = CIFARNET_QUICK(class_num = config.class_num, ctx=ctx)
net.load_params(params_file)

for block in net.layers:
    if not isinstance(block, CIFARNET_BLOCK):
        continue
    for layer in block.layers:
        if isinstance(layer, mx.gluon.nn.Conv2D):
            out_ch_num, in_ch_num, h, w = layer.weight.shape
            if h > vis_block_size or w > vis_block_size:
                print 'filter size should be less than ',vis_block_size
                continue
            vis = np.zeros( (out_ch_num * vis_block_size,in_ch_num * vis_block_size) )
            dx = (vis_block_size - w)//2
            dy = (vis_block_size - h)//2
            for out_idx in range(out_ch_num):
                for in_idx in range(in_ch_num):
                    data = layer.weight.data()[out_idx,:,:,:].asnumpy()
                    data = np.sum(data,axis=0) #input is gray image so here is ok
                    min,max = data.min(),data.max()
                    if max - min < 0.0001:
                        data = np.zeros((h,w),dtype=np.uint8)
                    else:
                        #data = np.abs(data)
                        #data = np.log(data - data.min() + 1)
                        #data = (data - min) / (max - min)
                       # data = np.log(data + 1)
                        data = (data - data.min()) * 255 / (data.max() - data.min())
                    x0, y0 = in_idx * vis_block_size + dx, out_idx * vis_block_size + dy
                    vis[y0:y0+data.shape[0], x0:x0+data.shape[1]] = data
            cv2.imwrite(layer.name + '.png',vis)


