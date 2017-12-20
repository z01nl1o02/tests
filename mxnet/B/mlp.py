import mxnet as mx
import logging
import gzip
import os,sys,pdb,pickle
datasetpath = 'mnist.pkl.gz'
with gzip.open(datasetpath,'rb') as f:
    trainset, validset, testset = pickle.load(f)
trainX = trainset[0]
trainY = trainset[1]
validX = validset[0]
validY = validset[1]
testX = testset[0]
testY = testset[1]

batch_size = 100
train_iter = mx.io.NDArrayIter(trainX,trainY,batch_size, shuffle=True)
val_iter = mx.io.NDArrayIter(validX,validY, batch_size)
test_iter = mx.io.NDArrayIter(testX, testY, batch_size)

data = mx.sym.var('data')
data = mx.sym.flatten(data=data)
fc1 = mx.sym.FullyConnected(data=data,num_hidden=128)
act1 = mx.sym.Activation(data=fc1, act_type='relu')

fc2 = mx.sym.FullyConnected(data=act1, num_hidden=64)
act2 = mx.sym.Activation(data=fc2,act_type="relu")

fc3 = mx.sym.FullyConnected(data=act2,num_hidden=10)
mlp = mx.sym.SoftmaxOutput(data=fc3,name="softmax")


logging.getLogger().setLevel(logging.DEBUG)

mlp_model = mx.mod.Module(symbol=mlp, context=mx.cpu())
mlp_model.fit(train_iter, eval_data=val_iter, optimizer='sgd',
optimizer_params={'learning_rate':0.1},
eval_metric='acc', batch_end_callback=mx.callback.Speedometer(batch_size,100),
num_epoch=10)


prob = mlp_model.predict(test_iter)
acc = mx.metric.Accuracy()
mlp_model.score( test_iter, acc)
print acc



