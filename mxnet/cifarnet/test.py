import os,sys,cv2
import mxnet as mx
from mxnet import gluon
from mxnet import nd
from mxnet.gluon import utils
import numpy as np
import logging
import random
from config import config
from symbol.cifarnet import CIFARNET_QUICK
from collections import defaultdict,OrderedDict
from tqdm import tqdm

list_file = 'list.txt'
params_file = 'models/used/cifarnet_epoch_{:0>5d}.params'.format(200)

ctx = mx.gpu()
crop_hw = (config.height, config.width)
net = CIFARNET_QUICK(class_num = config.class_num, ctx=ctx)
net.load_params(params_file)


path_list = []
with open(list_file,'rb') as f:
    for line in f:
        line = line.strip()
        if line == "":
            continue
        path_list.append(line)

lines = []
for path in tqdm(path_list):
    img = cv2.imread(path,1)
    H,W,C = img.shape
    dx = (W - crop_hw[1]) // 2
    dy = (H - crop_hw[0]) // 2
    img = img[dy:dy+crop_hw[0], dx:dx+crop_hw[1],:]
    input = np.transpose(np.float32(img), (2, 0, 1))
    input = np.expand_dims(input,0)
    input = nd.array(input).as_in_context(ctx)
    pred = net(input)
    pred = mx.nd.softmax(pred,axis=1).asnumpy()
    pred_label = np.argmax(pred,axis=1)[0]
    pred_score = pred.flatten()[pred_label]
    lines.append('{}|{:.3}|{}'.format(pred_label, pred_score, path))

with open('test_result.txt','wb') as f:
    f.write('\n'.join(lines))





