import mxnet as mx
import logging
import numpy as np
import os,sys,pdb,pickle

class COLORNET(object):
    def __init__(self):
        data = mx.sym.var('data')
        conv1 = mx.sym.Convolution(data=data,kernel=(5,5),num_filter=2)
        tanh1 = mx.sym.Activation(data=conv1,act_type='tanh')
        pool1 = mx.sym.Pooling(data=tanh1, pool_type='max', kernel=(2,2), stride = (2,2))

        do1 = mx.sym.Dropout(data=pool1,name="do1",p=0.25)

        conv2 = mx.sym.Convolution(data=do1, kernel=(5,5), num_filter=5)
        tanh2 = mx.sym.Activation(data=conv2, act_type='tanh')
        pool2 = mx.sym.Pooling(data=tanh2, pool_type='max', kernel = (2,2), stride=(2,2))

        do2 = mx.sym.Dropout(data=pool2,name="do2",p=0.25)

        flatten = mx.sym.flatten(data = do2)
        fc1 = mx.sym.FullyConnected(data=flatten, num_hidden=500)
        tanh3 = mx.sym.Activation(data=fc1,act_type='tanh')

        fc2 = mx.sym.FullyConnected(data=tanh3, num_hidden=13)
        symbol = mx.sym.SoftmaxOutput(data=fc2, name='softmax')
        self.net = mx.mod.Module(symbol=symbol, context=mx.cpu())
    def fit(self,train_iter, val_iter,batchsize): 
        try:
            os.makedirs('model')
        except Exception,e:
            pass
        head = '%(asctime)-15s %(message)s'
        logging.basicConfig(level=logging.DEBUG, format=head) #enable log out
        self.net.fit(train_iter, eval_data=val_iter, optimizer='sgd',
        optimizer_params={'learning_rate':0.1,'wd':0.00}, #weight decay
            eval_metric='acc',
            batch_end_callback=mx.callback.Speedometer(batchsize,100),
            epoch_end_callback=mx.callback.do_checkpoint("model\\colornaming"),
            num_epoch=2000)
    def predict(self,test_iter):
        prob = self.net.predict(test_iter)
        acc = mx.metric.Accuracy()
        self.net.score( test_iter, acc)
        return acc



