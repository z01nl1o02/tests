from mxnet import gluon, image
from mxnet.gluon import utils as gutils
import os
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import numpy as np
def _download_pikachu(data_dir):
    root_url = ('https://apache-mxnet.s3-accelerate.amazonaws.com/'
                'gluon/dataset/pikachu/')
    dataset = {'train.rec': 'e6bcb6ffba1ac04ff8a9b1115e650af56ee969c8',
               'train.idx': 'dcf7318b2602c06428b9988470c731621716c393',
               'val.rec': 'd6c33f799b4d058e82f2cb5bd9a976f69d72d520'}
    for k, v in dataset.items():
        gutils.download(root_url + k, os.path.join(data_dir, k), sha1_hash=v)
        
        
def load_data_pikachu(batch_size, edge_size=256):  # edge_size：输出图像的宽和高
    data_dir = 'data/pikachu'
    _download_pikachu(data_dir)
    train_iter = image.ImageDetIter(
        path_imgrec=os.path.join(data_dir, 'train.rec'),
        path_imgidx=os.path.join(data_dir, 'train.idx'),
        batch_size=batch_size,
        data_shape=(3, edge_size, edge_size),  # 输出图像的形状
        shuffle=False,  # 以随机顺序读取数据集
        rand_crop=0,  # 随机裁剪的概率为1
        min_object_covered=0.95, max_attempts=200)
    val_iter = image.ImageDetIter(
        path_imgrec=os.path.join(data_dir, 'val.rec'), batch_size=batch_size,
        data_shape=(3, edge_size, edge_size), shuffle=False)
    return train_iter, val_iter

batch_size = 1
train_iter, _ = load_data_pikachu(batch_size)



#from tqdm import tqdm 
lines = []
for k,batch in enumerate(train_iter):
    X,Y = batch.data[0].asnumpy(), batch.label[0].asnumpy()
    img = X[0].transpose((1,2,0)).astype(np.uint8)
    img = Image.fromarray(img)
    W,H = img.size
    rect = Y[0,0,1:] * np.asarray([W,H,W,H])
    #draw = ImageDraw.Draw(img)
    #draw.rectangle([(rect[0],rect[1]),(rect[2],rect[3])],None,255)
    w,h = rect[2] - rect[0], rect[3] - rect[1]
    print(w/W,',',h/H)
    filepath  = 'data/%d.jpg'%k
    line = [filepath]
    for j in range(Y.shape[1]):
        info = Y[0,j,:]
        line.append(' '.join(['%.3f'%d for d in info]))
    lines.append(' '.join(line))
    img.save(filepath)
    
with open('sample.txt','w') as f:
    f.write('\n'.join(lines))     
train_iter.reset()  # 从头读取数据   
