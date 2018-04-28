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
import random
import pdb,os,sys
import logging
import datetime

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
    return nd.sum(output.argmax(axis=1)==label).asscalar()
    #return nd.mean(output.argmax(axis=1)==label).asscalar()


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

def evaluate_accuracy(data_iterator, net, ctx=[mx.cpu()]):
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
            acc += nd.sum(net(X).argmax(axis=1)==y).copyto(mx.cpu())
            n += y.size
        acc.wait_to_read() # don't push too many operators into backend
    return acc.asscalar() / n

def save_checkpoints(cpdir,net, trainer,epoch):
    net.save_params('%s/epoch-%.6d.params'%(cpdir,epoch)) #with load_params() may regain most weight() but not all????
    #net.collect_params().save("yyy-%.4d.params"%i)
    #trainer.save_states('epoch-%.6d.state'%epoch) #load_params() error with kv_store???
    #trainer.load_states('zzz.state')
    
    
def train(train_data, valid_data, batch_size,\
        net, loss_func, trainer, ctx, num_epochs, \
        lr_period, lr_decay,print_batches=None, chk_pts_dir=None):
    print_freq = 100
    check_freq = 100
    if chk_pts_dir is not None:
        try:
            os.makedirs(chk_pts_dir)
        except Exception,e:
            print 'failed to build ',chk_pts_dir
    """Train a network"""
    logging.info("Start training on %s"%ctx)
    #if isinstance(ctx, mx.Context):
    #    ctx = [ctx]
    prev_time = datetime.datetime.now()
    for epoch in range(num_epochs):
        train_loss, train_acc, n, m = 0.0, 0.0, 0.0, 0.0
        if isinstance(train_data, mx.io.MXDataIter) or isinstance(train_data,mx.image.ImageIter):
            train_data.reset()
        if epoch > 0 and epoch % lr_period == 0:
            trainer.set_learning_rate( trainer.learning_rate * lr_decay )
        #pdb.set_trace()
        batchNum = 0
        trainNum = 0
        for datas, labels in train_data:
            #data, label, batch_size = _get_batch(batch, ctx)  
            #batch_size = batch.shape[0]
            #pdb.set_trace()
            trainNum += datas.asnumpy().shape[0]
            labels = labels.astype('float32').as_in_context(ctx) 
            with autograd.record(): #each sample each time
                yhats = net(datas.as_in_context(ctx))
                #losses = [ loss_func(yhat, label) for yhat,label in zip(yhats,labels)] 
                loss = loss_func(yhats,labels)
            #pdb.set_trace()
            #for loss in losses:
            #    loss.backward()
            loss.backward()
            trainer.step(batch_size)
            train_loss += nd.mean(loss).asscalar()
            train_acc += accuracy(yhats,labels)
            batchNum += 1
            cur_time = datetime.datetime.now()
            h, remainder = divmod((cur_time - prev_time).seconds, 3600)
            m, s = divmod(remainder, 60)
            time_str = "Time %02d:%02d:%02d" % (h, m, s)
            if valid_data is not None and 0 == (batchNum%check_freq):
                valid_acc = evaluate_accuracy(valid_data, net, ctx)
                epoch_str = ("Epoch %d. Batch %d Loss: %f, Train acc %f, Valid acc %f, "
                             % (epoch, batchNum, train_loss / trainNum,
                                train_acc / trainNum, valid_acc))
                logging.info(epoch_str + time_str + ', lr ' + str(trainer.learning_rate))
            elif 0 == (batchNum%print_freq):
                epoch_str = ("Epoch %d. Batch %d Loss: %f, Train acc %f, "
                             % (epoch, batchNum, train_loss / trainNum,
                                train_acc / trainNum))

                logging.info(epoch_str + time_str + ', lr ' + str(trainer.learning_rate))
            prev_time = cur_time
        if chk_pts_dir is not None:
            save_checkpoints(chk_pts_dir,net,trainer,epoch)

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


