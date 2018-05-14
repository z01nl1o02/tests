#--fileencoding=utf8---
import zipfile
import os,sys,pdb
import numpy as np
import random
import mxnet.ndarray as nd
from mxnet import gluon

with zipfile.ZipFile('jaychou_lyrics.txt.zip','r') as zin:
    zin.extractall('.')

with open('jaychou_lyrics.txt') as f:
    corpus_chars = f.read().decode('utf8')
print corpus_chars[0:49]

corpus_chars = corpus_chars.replace('\n',' ').replace('\r',' ')
corpus_chars = corpus_chars[0:20000]
#rint corpus_chars

idx_to_char = list( set(corpus_chars) )
char_to_idx = dict( [(char,idx) for idx, char in enumerate(idx_to_char)]  )
vocab_size = len(char_to_idx)
print 'vocab_size: ', vocab_size

#print char_to_idx[u'形']
#print '-', ''.join( [idx_to_char[k] for k in range(10)] ) 
#pdb.set_trace()

corpus_indices = [ char_to_idx[char] for char in corpus_chars ]
#pdb.set_trace()
def data_iter_random(corpus_indices, batch_size, num_steps, ctx=None):
    num_examples = len(corpus_indices)
    epoch_size = (num_examples - num_steps) // batch_size
    example_indices = list(range(num_examples - num_steps))
    random.shuffle(example_indices)


    def _data(pos):
        return corpus_indices[pos:pos + num_steps]

    for i in range(epoch_size):
        i = i * batch_size
        batch_indices = example_indices[i:i+batch_size]
        dat = [_data(j) for j in batch_indices]
        data = nd.array( dat, ctx=ctx)
        label = nd.array(
            [_data(j + 1) for j in batch_indices], ctx=ctx
        )
        yield data,label

my_seq = list(range(30))

for data, label in data_iter_random(my_seq, batch_size=2, num_steps=3):
    print('data: ', data, '\nlabel:', label, '\n')

def get_inputs(data):
    return [nd.one_hot(X, vocab_size) for X in data.T]





import mxnet as mx
import sys
sys.path.append('..')
import utils
ctx = utils.try_gpu()
ctx = mx.cpu() #gpu bug about (-1)**2 to be nan!!
print('Will use', ctx)

input_dim = vocab_size
hidden_dim = 256
output_dim = vocab_size
std = .01

def get_params():
    W_xh = nd.random_normal(scale=std, shape=(input_dim, hidden_dim), ctx=ctx)
    W_hh = nd.random_normal(scale=std, shape=(hidden_dim, hidden_dim), ctx=ctx)
    b_h = nd.zeros(hidden_dim, ctx=ctx)

    W_hy = nd.random_normal(scale=std, shape=(hidden_dim, output_dim), ctx=ctx)
    b_y = nd.zeros(output_dim, ctx=ctx)

    params = [W_xh, W_hh, b_h, W_hy, b_y]
    for param in params:
        param.attach_grad()
    return params



def rnn(inputs, state, *params):
    H = state
    W_xh, W_hh, b_h, W_hy, b_y = params
    outputs = []
    for X in inputs:
        H = nd.tanh(nd.dot(X, W_xh) + nd.dot(H, W_hh) + b_h)
        Y = nd.dot(H, W_hy) + b_y
        outputs.append(Y)
    return (outputs, H)



state = nd.zeros(shape=(data.shape[0], hidden_dim), ctx=ctx)

params = get_params()
outputs, state_new = rnn(get_inputs(data.as_in_context(ctx)), state, *params)

print('output length: ',len(outputs))
print('output[0] shape: ', outputs[0].shape)
print('state shape: ', state_new.shape)


def predict_rnn(rnn, prefix, num_chars, params, hidden_dim, ctx, idx_to_char,
                char_to_idx, get_inputs, is_lstm=False):

    prefix = prefix.lower()
    state_h = nd.zeros(shape=(1, hidden_dim), ctx=ctx)
    if is_lstm:
        state_c = nd.zeros(shape=(1, hidden_dim), ctx=ctx)
    #pdb.set_trace()
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


epochs = 200
num_steps = 35
learning_rate = 0.1
batch_size = 32

softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()

seq1 = u'分开'
seq2 = u'不分开'
seq3 = u'战斗中的部队'
seqs = [seq1, seq2, seq3]

def grad_clipping(params, theta, ctx):
    if theta is not None:
        norm = nd.array([0.0], ctx)
        for p in params:
            norm += nd.sum(p.grad * p.grad)
        norm = nd.sqrt(norm).asscalar()
        if norm > theta:
            for p in params:
                p.grad[:] *= theta / norm

from mxnet import autograd
from mxnet import gluon
from math import exp

def train_and_predict_rnn(rnn, is_random_iter, epochs, num_steps, hidden_dim,
                          learning_rate, clipping_theta, batch_size,
                          pred_period, pred_len, seqs, get_params, get_inputs,
                          ctx, corpus_indices, idx_to_char, char_to_idx,
                          is_lstm=False):
    if is_random_iter:
        data_iter = data_iter_random
    else:
        data_iter = data_iter_consecutive
    params = get_params()

    softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()

    for e in range(1, epochs + 1):
        # 如使用相邻批量采样，在同一个epoch中，隐含变量只需要在该epoch开始的时候初始化。
        if not is_random_iter:
            state_h = nd.zeros(shape=(batch_size, hidden_dim), ctx=ctx)
            if is_lstm:
                # 当RNN使用LSTM时才会用到，这里可以忽略。
                state_c = nd.zeros(shape=(batch_size, hidden_dim), ctx=ctx)
        train_loss, num_examples = 0, 0
        for data, label in data_iter(corpus_indices, batch_size, num_steps,
                                     ctx):
            # 如使用随机批量采样，处理每个随机小批量前都需要初始化隐含变量。
            if is_random_iter:
                state_h = nd.zeros(shape=(batch_size, hidden_dim), ctx=ctx)
                if is_lstm:
                    # 当RNN使用LSTM时才会用到，这里可以忽略。
                    state_c = nd.zeros(shape=(batch_size, hidden_dim), ctx=ctx)
            with autograd.record():
                # outputs 尺寸：(batch_size, vocab_size)
                if is_lstm:
                    # 当RNN使用LSTM时才会用到，这里可以忽略。
                    outputs, state_h, state_c = rnn(get_inputs(data), state_h,
                                                    state_c, *params)
                else:
                    outputs, state_h = rnn(get_inputs(data), state_h, *params)
                # 设t_ib_j为i时间批量中的j元素:
                # label 尺寸：（batch_size * num_steps）
                # label = [t_0b_0, t_0b_1, ..., t_1b_0, t_1b_1, ..., ]
                label = label.T.reshape((-1,))
                # 拼接outputs，尺寸：(batch_size * num_steps, vocab_size)。
                outputs = nd.concat(*outputs, dim=0)
                # 经上述操作，outputs和label已对齐。
                loss = softmax_cross_entropy(outputs, label)
            loss.backward()

            grad_clipping(params, clipping_theta, ctx)
            utils.SGD(params, learning_rate)

            train_loss += nd.sum(loss).asscalar()
            num_examples += loss.size

        if e % pred_period == 0:
            print("Epoch %d. Perplexity %f" % (e,
                                               exp(train_loss/num_examples)))
            for seq in seqs:
                print' - ', predict_rnn(rnn, seq, pred_len, params,
                      hidden_dim, ctx, idx_to_char, char_to_idx, get_inputs,
                      is_lstm)
            print()

train_and_predict_rnn(rnn=rnn, is_random_iter=True, epochs=200, num_steps=35,
                      hidden_dim=hidden_dim, learning_rate=0.2,
                      clipping_theta=5, batch_size=32, pred_period=1,
                      pred_len=100, seqs=seqs, get_params=get_params,
                      get_inputs=get_inputs, ctx=ctx,
                      corpus_indices=corpus_indices, idx_to_char=idx_to_char,
                      char_to_idx=char_to_idx)
