from math import exp
from mxnet import gluon
from mxnet import autograd
from mxnet import nd
from mxnet import image
from mxnet.gluon import nn
import mxnet as mx
import numpy as np
from time import time
import matplotlib.pyplot as plt
import matplotlib as mpl
import random,cPickle
import pdb,os,sys
import logging

class DataLoader(object):
    """similiar to gluon.data.DataLoader, but might be faster.

    The main difference this data loader tries to read more exmaples each
    time. But the limits are 1) all examples in dataset have the same shape, 2)
    data transfomer needs to process multiple examples at each time
    """
    def __init__(self, dataset, batch_size, shuffle, transform=None):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.transform = transform

    def __iter__(self):
        data = self.dataset[:]
        X = data[0]
        y = nd.array(data[1])
        n = X.shape[0]
        if self.shuffle:
            idx = np.arange(n)
            np.random.shuffle(idx)
            X = nd.array(X.asnumpy()[idx])
            y = nd.array(y.asnumpy()[idx])

        for i in range(n//self.batch_size):
            if self.transform is not None:
                yield self.transform(X[i*self.batch_size:(i+1)*self.batch_size], 
                                     y[i*self.batch_size:(i+1)*self.batch_size])
            else:
                yield (X[i*self.batch_size:(i+1)*self.batch_size],
                       y[i*self.batch_size:(i+1)*self.batch_size])

    def __len__(self):
        return len(self.dataset)//self.batch_size

def load_data_fashion_mnist(batch_size, resize=None, root="~/.mxnet/datasets/fashion-mnist"):
    """download the fashion mnist dataest and then load into memory"""
    def transform_mnist(data, label):
        # Transform a batch of examples.
        if resize:
            n = data.shape[0]
            new_data = nd.zeros((n, resize, resize, data.shape[3]))
            for i in range(n):
                new_data[i] = image.imresize(data[i], resize, resize)
            data = new_data
        # change data from batch x height x width x channel to batch x channel x height x width
        return nd.transpose(data.astype('float32'), (0,3,1,2))/255, label.astype('float32')

    mnist_train = gluon.data.vision.FashionMNIST(root=root, train=True, transform=None)
    mnist_test = gluon.data.vision.FashionMNIST(root=root, train=False, transform=None)
    # Transform later to avoid memory explosion. 
    train_data = DataLoader(mnist_train, batch_size, shuffle=True, transform=transform_mnist)
    test_data = DataLoader(mnist_test, batch_size, shuffle=False, transform=transform_mnist)
    return (train_data, test_data)

def try_gpu():
    """If GPU is available, return mx.gpu(0); else return mx.cpu()"""
    try:
        ctx = mx.gpu()
        _ = nd.array([0], ctx=ctx)
    except:
        ctx = mx.cpu()
    return ctx

def try_all_gpus():
    """Return all available GPUs, or [mx.gpu()] if there is no GPU"""
    ctx_list = []
    try:
        for i in range(16):
            ctx = mx.gpu(i)
            _ = nd.array([0], ctx=ctx)
            ctx_list.append(ctx)
    except:
        pass
    if not ctx_list:
        ctx_list = [mx.cpu()]
    return ctx_list

def SGD(params, lr):
    for param in params:
        param[:] = param - lr * param.grad

def accuracy(output, label):
    return nd.mean(output.argmax(axis=1)==label).asscalar()


def _get_batch(batch, ctx):
    """return data and label on ctx"""
    if isinstance(batch, mx.io.DataBatch):
        data = batch.data[0]
        label = batch.label[0]
    else:
        data, label = batch
    return (gluon.utils.split_and_load(data, ctx),
            gluon.utils.split_and_load(label, ctx),
            data.shape[0])

def evaluate_loss(data_iterator, net, ctx=[mx.cpu()]):
    if isinstance(ctx, mx.Context):
        ctx = [ctx]
    acc = nd.array([0])
    n = 0.
    if isinstance(data_iterator, mx.io.MXDataIter) or isinstance(data_iterator,mx.image.ImageIter):
        data_iterator.reset()
    for batch in data_iterator:
        data, label, batch_size = _get_batch(batch, ctx)
        for X, y in zip(data, label):
            y = y.astype('float32')
            y0 = net(X)
            #acc += nd.sum( (y0-y)*(y0-y) ).copyto(mx.cpu())
            acc += nd.mean( 0.5*(y0-y)*(y0-y), axis=1 ).copyto(mx.cpu()).sum() #mean along dim of L2Loss
            n += y.shape[0]
        acc.wait_to_read() # don't push too many operators into backend
    
    return acc.asscalar() / n #mean of L2Loss

def save_checkpoints(cpdir,net, trainer,epoch):
    net.save_params('%s/epoch-%.12d.params'%(cpdir,epoch)) #with load_params() may regain most weight() but not all????
    #net.collect_params().save("yyy-%.4d.params"%i)
    #trainer.save_states('epoch-%.6d.state'%epoch) #load_params() error with kv_store???
    #trainer.load_states('zzz.state')
            
def train(train_data, test_data, net, lossfunc, trainer, ctx, num_epochs, lr_steps, print_batches=None, cpdir=None):
    if cpdir is not None:
        try:
            os.makedirs(cpdir)
        except Exception,e:
            pass
    """Train a network"""
    logging.info("Start training on %s"%ctx)
    if isinstance(ctx, mx.Context):
        ctx = [ctx]
    loss_train = []
    loss_test = []
    lr_train = []
    iter = 0
    for epoch in range(num_epochs):
        train_loss = 0.0
        if isinstance(train_data, mx.io.MXDataIter) or isinstance(train_data,mx.image.ImageIter):
            train_data.reset()
        
        start = time()
        n = 0
        for batch in train_data:
            iter += 1
            
            if iter in set( lr_steps ):
                logging.info("#%d update lr: %f"%(iter, trainer.learning_rate * 0.1))
                trainer.set_learning_rate( trainer.learning_rate * 0.1 )
                lr_train.append( (iter, trainer.learning_rate * 0.1)  )
            data, label, batch_size = _get_batch(batch, ctx)  
            #batch_size = batch.shape[0]
            losses = []
            with autograd.record():
                outputs = [net(X) for X in data]
                losses = [ lossfunc(yhat , y) for yhat, y in zip(outputs, label)]
            for l in losses:
                l.backward()
            n += batch_size
            train_loss += sum([l.sum().asscalar() for l in losses])
            #train_loss *= label[0].shape[1] #mxnet L2Loss is mean of L2Loss !!
            trainer.step(batch_size)
            loss_train.append( (iter,train_loss / n) )
            if print_batches and iter % print_batches == 0:
                logging.info("#%d lr %lf, Batch %d Train Loss: %lf" % (iter, trainer.learning_rate, n, train_loss/n))

        test_loss = evaluate_loss(test_data, net, ctx)
        loss_test.append((iter,test_loss))
        logging.info("Epoch %d. Train Loss: %lf, Test Loss %lf, Time %.1f sec" % (
            epoch, train_loss/n, test_loss, time() - start
        ))
        with open( os.path.join(cpdir,'loss.pkl'), "wb") as f:
            cPickle.dump((loss_train,loss_test,lr_train),f)
            
        if cpdir is not None:
            save_checkpoints(cpdir,net,trainer,iter)

class Residual(nn.HybridBlock):
    def __init__(self, channels, same_shape=True, **kwargs):
        super(Residual, self).__init__(**kwargs)
        self.same_shape = same_shape
        with self.name_scope():
            strides = 1 if same_shape else 2
            self.conv1 = nn.Conv2D(channels, kernel_size=3, padding=1,
                                  strides=strides)
            self.bn1 = nn.BatchNorm()
            self.conv2 = nn.Conv2D(channels, kernel_size=3, padding=1)
            self.bn2 = nn.BatchNorm()
            if not same_shape:
                self.conv3 = nn.Conv2D(channels, kernel_size=1,
                                      strides=strides)

    def hybrid_forward(self, F, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if not self.same_shape:
            x = self.conv3(x)
        return F.relu(out + x)

def resnet18(num_classes):
    net = nn.HybridSequential()
    with net.name_scope():
        net.add(
            nn.BatchNorm(),
            nn.Conv2D(64, kernel_size=3, strides=1),
            nn.MaxPool2D(pool_size=3, strides=2),
            Residual(64),
            Residual(64),
            Residual(128, same_shape=False),
            Residual(128),
            Residual(256, same_shape=False),
            Residual(256),
            nn.GlobalAvgPool2D(),
            nn.Dense(num_classes)
        )
    return net

def show_images(imgs, nrows, ncols, figsize=None):
    """plot a list of images"""
    if not figsize:
        figsize = (ncols, nrows)
    _, figs = plt.subplots(nrows, ncols, figsize=figsize)
    for i in range(nrows):
        for j in range(ncols):
            figs[i][j].imshow(imgs[i*ncols+j].asnumpy())
            figs[i][j].axes.get_xaxis().set_visible(False)
            figs[i][j].axes.get_yaxis().set_visible(False)
    plt.show()

def data_iter_random(corpus_indices, batch_size, num_steps, ctx=None):
    """Sample mini-batches in a random order from sequential data."""
    # Subtract 1 because label indices are corresponding input indices + 1. 
    num_examples = (len(corpus_indices) - 1) // num_steps
    epoch_size = num_examples // batch_size
    # Randomize samples.
    example_indices = list(range(num_examples))
    random.shuffle(example_indices)

    def _data(pos):
        return corpus_indices[pos: pos + num_steps]

    for i in range(epoch_size):
        # Read batch_size random samples each time.
        i = i * batch_size
        batch_indices = example_indices[i: i + batch_size]
        data = nd.array(
            [_data(j * num_steps) for j in batch_indices], ctx=ctx)
        label = nd.array(
            [_data(j * num_steps + 1) for j in batch_indices], ctx=ctx)
        yield data, label

def data_iter_consecutive(corpus_indices, batch_size, num_steps, ctx=None):
    """Sample mini-batches in a consecutive order from sequential data."""
    corpus_indices = nd.array(corpus_indices, ctx=ctx)
    data_len = len(corpus_indices)
    batch_len = data_len // batch_size
    
    indices = corpus_indices[0: batch_size * batch_len].reshape((
        batch_size, batch_len))
    # Subtract 1 because label indices are corresponding input indices + 1. 
    epoch_size = (batch_len - 1) // num_steps
    
    for i in range(epoch_size):
        i = i * num_steps
        data = indices[:, i: i + num_steps]
        label = indices[:, i + 1: i + num_steps + 1]
        yield data, label


def grad_clipping(params, clipping_norm, ctx):
    """Gradient clipping."""
    if clipping_norm is not None:
        norm = nd.array([0.0], ctx)
        for p in params:
            norm += nd.sum(p.grad ** 2)
        norm = nd.sqrt(norm).asscalar()
        if norm > clipping_norm:
            for p in params:
                p.grad[:] *= clipping_norm / norm


def predict_rnn(rnn, prefix, num_chars, params, hidden_dim, ctx, idx_to_char,
                char_to_idx, get_inputs, is_lstm=False):
    """Predict the next chars given the prefix."""
    prefix = prefix.lower()
    state_h = nd.zeros(shape=(1, hidden_dim), ctx=ctx)
    if is_lstm:
        state_c = nd.zeros(shape=(1, hidden_dim), ctx=ctx)
    output = [char_to_idx[prefix[0]]]
    for i in range(num_chars + len(prefix)):
        X = nd.array([output[-1]], ctx=ctx)
        if is_lstm:
            Y, state_h, state_c = rnn(get_inputs(X), state_h, state_c, *params)
        else:
            Y, state_h = rnn(get_inputs(X), state_h, *params)
        if i < len(prefix)-1:
            next_input = char_to_idx[prefix[i+1]]
        else:
            next_input = int(Y[0].argmax(axis=1).asscalar())
        output.append(next_input)
    return ''.join([idx_to_char[i] for i in output])


def train_and_predict_rnn(rnn, is_random_iter, epochs, num_steps, hidden_dim, 
                          learning_rate, clipping_norm, batch_size,
                          pred_period, pred_len, seqs, get_params, get_inputs,
                          ctx, corpus_indices, idx_to_char, char_to_idx,
                          is_lstm=False):
    """Train an RNN model and predict the next item in the sequence."""
    if is_random_iter:
        data_iter = data_iter_random
    else:
        data_iter = data_iter_consecutive
    params = get_params()
    
    softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()

    for e in range(1, epochs + 1): 
        # If consecutive sampling is used, in the same epoch, the hidden state
        # is initialized only at the beginning of the epoch.
        if not is_random_iter:
            state_h = nd.zeros(shape=(batch_size, hidden_dim), ctx=ctx)
            if is_lstm:
                state_c = nd.zeros(shape=(batch_size, hidden_dim), ctx=ctx)
        train_loss, num_examples = 0, 0
        for data, label in data_iter(corpus_indices, batch_size, num_steps, 
                                     ctx):
            # If random sampling is used, the hidden state has to be
            # initialized for each mini-batch.
            if is_random_iter:
                state_h = nd.zeros(shape=(batch_size, hidden_dim), ctx=ctx)
                if is_lstm:
                    state_c = nd.zeros(shape=(batch_size, hidden_dim), ctx=ctx)
            with autograd.record():
                # outputs shape: (batch_size, vocab_size)
                if is_lstm:
                    outputs, state_h, state_c = rnn(get_inputs(data), state_h,
                                                    state_c, *params) 
                else:
                    outputs, state_h = rnn(get_inputs(data), state_h, *params)
                # Let t_ib_j be the j-th element of the mini-batch at time i.
                # label shape: (batch_size * num_steps)
                # label = [t_0b_0, t_0b_1, ..., t_1b_0, t_1b_1, ..., ].
                label = label.T.reshape((-1,))
                # Concatenate outputs:
                # shape: (batch_size * num_steps, vocab_size).
                outputs = nd.concat(*outputs, dim=0)
                # Now outputs and label are aligned.
                loss = softmax_cross_entropy(outputs, label)
            loss.backward()

            grad_clipping(params, clipping_norm, ctx)
            SGD(params, learning_rate)

            train_loss += nd.sum(loss).asscalar()
            num_examples += loss.size

        if e % pred_period == 0:
            print("Epoch %d. Training perplexity %f" % (e, 
                                               exp(train_loss/num_examples)))
            for seq in seqs:
                print(' - ', predict_rnn(rnn, seq, pred_len, params,
                      hidden_dim, ctx, idx_to_char, char_to_idx, get_inputs,
                      is_lstm))
            print()

def set_fig_size(mpl, figsize=(3.5, 2.5)):
    """set output image size for matplotlib """
    mpl.rcParams['figure.figsize'] = figsize


def data_iter(batch_size, num_examples, X, y):
    """walk around dataset"""
    idx = list(range(num_examples))
    random.shuffle(idx)
    for i in range(0, num_examples, batch_size):
        j = nd.array(idx[i: min(i + batch_size, num_examples)])
        yield X.take(j), y.take(j)


def linreg(X, w, b):
    """linear regression"""
    return nd.dot(X, w) + b


def squared_loss(yhat, y):
    return (yhat - y.reshape(yhat.shape)) ** 2 / 2


def optimize(batch_size, trainer, num_epochs, decay_epoch, log_interval, X, y,
             net):
    dataset = gluon.data.ArrayDataset(X, y)
    data_iter = gluon.data.DataLoader(dataset, batch_size, shuffle=True)
    square_loss = gluon.loss.L2Loss()
    y_vals = [square_loss(net(X), y).mean().asnumpy()]
    for epoch in range(1, num_epochs + 1): 
        #lower lr
        if decay_epoch and epoch > decay_epoch:
            trainer.set_learning_rate(trainer.learning_rate * 0.1)
        for batch_i, (features, label) in enumerate(data_iter):
            with autograd.record():
                output = net(features)
                loss = square_loss(output, label)
            loss.backward()
            trainer.step(batch_size)
            if batch_i * batch_size % log_interval == 0:
                y_vals.append(square_loss(net(X), y).mean().asnumpy())
    print('w:', net[0].weight.data(), '\nb:', net[0].bias.data(), '\n')
    x_vals = np.linspace(0, num_epochs, len(y_vals), endpoint=True)
    semilogy(x_vals, y_vals, 'epoch', 'loss')


def semilogy(x_vals, y_vals, x_label, y_label, figsize=(3.5, 2.5)):
    """plot log(y)"""		
    set_fig_size(mpl, figsize)
    plt.semilogy(x_vals, y_vals)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()


