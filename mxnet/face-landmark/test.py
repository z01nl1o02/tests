from mxnet import image
from mxnet import nd
import mxnet as mx
import pdb
import utils,os
from importlib import import_module
import numpy as np
trainBatchSize = 2
testBatchSize = 2
outputNum = 10
data_dir = './'
data_shape = 64
batch_size = 2
path_root="c:/dataset/landmark/train/for-mxnet/train/"
dataShape = (3,64,64)

mean = [123.68, 116.28, 103.53]
std = [58.395, 57.12, 57.375]

def get_train_test():
    aug = mx.image.CreateAugmenter(data_shape = dataShape, resize = dataShape[1], mean=True, std=True)

    trainIter = mx.image.ImageIter(batch_size=trainBatchSize, data_shape=dataShape, 
                label_width = outputNum,
                path_imglist=os.path.join(path_root,'landmarks.lst'),
                path_root=path_root,
                aug_list=aug)
    testIter = mx.image.ImageIter(batch_size=trainBatchSize, data_shape=dataShape, 
                label_width = outputNum,
                path_imglist=os.path.join(path_root,'landmarks.lst'),
                path_root=path_root,
                aug_list=aug)
    return (trainIter, testIter)

train_data,test_iter = get_train_test()
ctx=mx.cpu()
     
mod = import_module('symbol.convnet_tanh')
net = mod.get_symbol(outputNum,ctx)


net.load_params("cp/epoch-000048.params",ctx=ctx)    


#%matplotlib inline
import matplotlib as mpl
mpl.rcParams['figure.dpi']= 120
import matplotlib.pyplot as plt
import cv2
import numpy as np
n = 0
acc = nd.array([0])
for batch in train_data:
    #pdb.set_trace()
    data, label, batch_size = utils._get_batch(batch, [ctx])
    for X, y in zip(data, label):
        y = y.astype('float32')
        y0 = net(X)
        acc += nd.sum( (y0-y)*(y0-y) ).copyto(mx.cpu())
        n += y.shape[0]
        #pdb.set_trace()
        print 'n = %d acc = %f'%(n,acc.asscalar() / n)
"""       
        img = X.asnumpy()[0]
        for k in range(3):
            img[k,:,:] = img[k,:,:] * std[k] + mean[k] #restore mean/std

         
        img = np.transpose(img,(1,2,0))
        img = cv2.cvtColor( np.uint8(img), cv2.COLOR_BGR2GRAY)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        img = cv2.resize(img, (img.shape[1] * 1, img.shape[0]*1))
        for k in range(0,10,2):
            #pdb.set_trace()
            x,y = y0.asnumpy()[0,k],y0.asnumpy()[0,k+1]
            x,y = np.int64(x * img.shape[1]), np.int64(y * img.shape[0])
            cv2.circle(img, (x,y), 3,(128,255,128))
         
        cv2.imshow("src",np.uint8(img))
        cv2.waitKey(-1)
"""
        
        #print y0.asnumpy()[0,:]
acc.wait_to_read()
print 'total ',n
print 'acc = ',acc.asscalar() / (2*n)
