import mxnet as mx
import logging
import numpy as np
import gzip
import os,sys,pdb,pickle

def cvt_to_4d(X):
    num = len(X)
    w,h = (28,28)
    c = 1
    result = np.zeros( (num, c, w, h), dtype=np.float32)
    for k in range(num):
        result[k,0,:,:] = np.reshape( X[k], (h,w))
    return result

datasetpath = 'mnist.pkl.gz'
with gzip.open(datasetpath,'rb') as f:
    trainset, validset, testset = pickle.load(f)
trainX = cvt_to_4d( trainset[0] )
trainY = trainset[1]
validX = cvt_to_4d( validset[0] )
validY = validset[1]
testX = cvt_to_4d( testset[0] )
testY = testset[1]

batch_size = 100
train_iter = mx.io.NDArrayIter(trainX,trainY,batch_size, shuffle=True)
val_iter = mx.io.NDArrayIter(validX,validY, batch_size)
test_iter = mx.io.NDArrayIter(testX, testY, batch_size)

data = mx.sym.var('data')
conv1 = mx.sym.Convolution(data=data,kernel=(5,5),num_filter=20)
tanh1 = mx.sym.Activation(data=conv1,act_type='tanh')
pool1 = mx.sym.Pooling(data=tanh1, pool_type='max', kernel=(2,2), stride = (2,2))

conv2 = mx.sym.Convolution(data=pool1, kernel=(5,5), num_filter=50)
tanh2 = mx.sym.Activation(data=conv2, act_type='tanh')
pool2 = mx.sym.Pooling(data=tanh2, pool_type='max', kernel = (2,2), stride=(2,2))

flatten = mx.sym.flatten(data = pool2)
fc1 = mx.sym.FullyConnected(data=flatten, num_hidden=500)
tanh3 = mx.sym.Activation(data=fc1,act_type='tanh')

fc2 = mx.sym.FullyConnected(data=tanh3, num_hidden=10)
lenet = mx.sym.SoftmaxOutput(data=fc2, name='softmax')

lenet_model = mx.mod.Module(symbol=lenet, context=mx.cpu())
lenet_model.fit(train_iter, eval_data=val_iter, optimizer='sgd',
optimizer_params={'learning_rate':0.1},
eval_metric='acc', 
batch_end_callback=mx.callback.Speedometer(batch_size,100),
num_epoch=10)


prob = lenet_model.predict(test_iter)
acc = mx.metric.Accuracy()
lenet_model.score( test_iter, acc)
print acc



