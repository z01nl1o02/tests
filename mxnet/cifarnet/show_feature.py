import os,sys
import cv2
import mxnet as mx
from mxnet import nd
import numpy as np
from config import config
from symbol.cifarnet import CIFARNET_QUICK,CIFARNET_BLOCK
from tqdm import tqdm

params_file = 'models/used/cifarnet_epoch_{:0>5d}.params'.format(200)
#img_file = "C:/dataset/mnist/test/0/3959.jpg"
img_file = "neg.bmp"
ctx = mx.cpu()
crop_hw = (config.height, config.width)
net = CIFARNET_QUICK(class_num = 2, ctx=ctx)
net.load_params(params_file)


crop_hw = (64,64)
img = cv2.imread(img_file,1)
H,W,C = img.shape
dx = (W - crop_hw[1]) // 2
dy = (H - crop_hw[0]) // 2
img = img[dy:dy+crop_hw[0], dx:dx+crop_hw[1],:]

if 1:
    out = np.transpose(img,(2,0,1))
    out = np.expand_dims(out, 0)
    out = np.float32(out)
else:
    out = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    out = np.expand_dims(out, 0)
    out = np.tile(out, (3, 1, 1))
    out = np.expand_dims(out,0)
    out = np.float32(out)

out = nd.array(out)
for block in net.layers:
    if not isinstance(block, CIFARNET_BLOCK):
        out = block(out)
        continue

    for layer in block.layers:
        if not isinstance(layer, mx.gluon.nn.Conv2D):
            out = layer(out)
            continue
        out = layer(out)
        data = out.asnumpy()
        B,C,H,W = data.shape
        assert(B == 1)
        vis_w = C // 5
        if vis_w < 1:
            vis_w = C
            vis_h = 1
        else:
            vis_h = (C + vis_w - 1) / vis_w
        vis = np.zeros((vis_h * H, vis_w*W))
        for y in range(vis_h):
            for x in range(vis_w):
                c = y * vis_w + x
                if c >= C:
                    break
                plane = data[0,c]
                plane = (plane - plane.min()) * 255 / (plane.max() - plane.min())
                vis[y*H:(y+1)*H,x*W:(x+1)*W] = plane
        vis = np.uint8(vis)
        cv2.imwrite(layer.name + ".png",vis)

