import os,sys,pdb,pickle
import numpy as np
import cv2
import theano
import theano.tensor as T
import random
#you may try several times to get a good model and the init 'cost' may be quite large, 78 .e.g.
class Layer(object):
    """
    a layer is a maxtrix with row = output of this layer and col = output of 
    previous layer
    """
    def __init__(self, W_init, b_init, activation):
        n_output,n_input = W_init.shape
        assert b_init.shape == (n_output,)
        self.W = theano.shared(value=W_init.astype(theano.config.floatX),
                                name="W",
                                borrow=True)
        self.b = theano.shared(value=b_init.reshape(n_output,1).astype(theano.config.floatX),
                                name="b",
                                borrow=True,
                                broadcastable=(False,True))
        self.activation = activation
        self.params = [self.W, self.b]

    def output(self,x):
        lin_output = T.dot(self.W,x) + self.b
        return (lin_output if self.activation is None else self.activation(lin_output))

class MLP(object):
    def __init__(self, W_init, b_init, activations):
        assert len(W_init) == len(b_init) == len(activations)
        self.layers = []
        for W,b,activation in zip(W_init, b_init, activations):
            self.layers.append(Layer(W,b,activation))
        self.params = []
        for layer in self.layers:
            self.params += layer.params

    def output(self,x):
        for layer in self.layers:
            x = layer.output(x)
        return x

    def squared_error(self,x,y):
         return T.sum((self.output(x) - y) ** 2)
         return T.mean((self.output(x) - y) ** 2)

    def cvt2c(self):
        line = ""
        for param in self.params:
            parval = param.get_value()
            line += "%d"%(parval.shape[0]) + ',' + "%d"%(parval.shape[1]) + '\n'
            for y in range(parval.shape[0]):
                for x in range(parval.shape[1]):
                    line += "%lf"%(parval[y,x])+ ','
            line += '\n'       
        return line
             

class MLP_PROXY(object): 
    def __init__(self, modelpath):
        self._train = None
        self._predict = None
        self._cost = None
        self._minmax = None
        self._modelpath = modelpath
        self._mlp = None
    def gradient_updates_momentum(self,cost, params, learning_rate, momentum):
        assert momentum < 1 and momentum >= 0
        updates = []
        for param in params:
            param_update = theano.shared(param.get_value() * 0., broadcastable=param.broadcastable)
            updates.append((param, param - learning_rate * param_update))
            updates.append((param_update, momentum * param_update + (1. - momentum)*T.grad(cost, param)))
        return updates

    def write_in_c_format(self,outpath):
        line = ""
        for m0,m1 in zip(self._minmax[0], self._minmax[1]):
            line += "%lf,%lf,"%(m0,m1)  
        line += '\n'
        line += self._mlp.cvt2c()
        with open(outpath, 'w') as f:
            f.writelines(line)
        return

    def create(self, layer_sizes, learning_rate = 0.01, momentum = 0.6):
        W_init = []
        b_init = []
        activations = []
        for n_input, n_output in zip(layer_sizes[:-1], layer_sizes[1:]):
            W_init.append(np.random.randn(n_output, n_input))
            b_init.append(np.random.randn(n_output))
            activations.append(T.nnet.sigmoid)
        mlp = MLP(W_init, b_init, activations)
        mlp_input = T.matrix('mlp_input')
        mlp_target = T.matrix('mlp_target')
        self._cost = mlp.squared_error(mlp_input, mlp_target)
        self._train = theano.function([mlp_input,mlp_target], self._cost, updates=self.gradient_updates_momentum(self._cost, mlp.params, learning_rate, momentum))
        self._predict = theano.function([mlp_input],  mlp.output(mlp_input))
        self._mlp = mlp
        return

    def train(self,samples, targets, max_iteration=5000, min_cost = 0.01):
        #samples and targets : (samples num) X (feature dimenstion)
        iteration = 0
        samplesT = np.transpose(samples) #W*x + b
        targetsT = np.transpose(targets)
        batchsize = 5
        echostep = max_iteration / 10
        if echostep > 1000:
            echostep = 1000
        while iteration < max_iteration:
            cost = 0
            total = 0
            for k in range(0,samplesT.shape[1],batchsize):
                kk = k
                if kk + batchsize > samplesT.shape[1]:
                    kk = samplesT.shape[1] - batchsize
                s = np.reshape(samplesT[:,kk:kk+batchsize],(-1,batchsize))
                t = np.reshape(targetsT[:,kk:kk+batchsize],(-1,batchsize))
                current_cost = self._train(s,t)
                cost =  cost + current_cost.sum()
                total += batchsize
            
            if (1+iteration)% echostep == 0: 
                print iteration + 1, ',', cost
            if cost < min_cost:
                break
            iteration += 1
        return
    def predict(self,samples):
        samplesT = np.transpose(samples) #W*x + b
        output = self._predict(samplesT)
        targets = np.transpose(output)
        return targets

    def pre_normalization(self, samples):
        m0 = samples[0,:]
        m1 = samples[0,:]
        for k in range(1,samples.shape[0]):
            m0 = np.minimum(samples[k,:],m0)
            m1 = np.maximum(samples[k,:],m1)
        self._minmax = (m0,m1)
        return

    def normalization(self, samples, u=1, l=-1):
        if None == self._minmax:
            return None
        m0,m1 = self._minmax
        rng = m1 - m0
        tmp = np.ones(rng.shape)
        for k in range(len(rng)):
            if rng[k] < 0.001:
                rng[k] = 1
                tmp[k] = 0
        ratio = tmp / rng
        for k in range(samples.shape[0]):
            feat = samples[k,:]
            feat = (feat - m0) * ratio * (u - l) + l
            idx = feat>u
            feat[idx] = u
            idx = feat<l
            feat[idx] = l
            samples[k,:] = feat
        return samples
        
    def shuffle(self, samples, targets):
       totalnum = samples.shape[0]
       idx = range(totalnum)
       random.shuffle(idx)
       rnd_samples = np.zeros(samples.shape)
       rnd_targets = np.zeros(targets.shape)
       for k in range(len(idx)):
           i = idx[k]
           rnd_samples[k,:] = samples[i,:] 
           rnd_targets[k,:] = targets[i,:] 
       return (rnd_samples, rnd_targets)
         
    def target_vec2mat(self, target_list, labelnum, hvalue = 1.0, lvalue = 0.0):
        #0-based
        targetnum = len(target_list)
        targets = np.zeros((targetnum, labelnum))
        for k in range(targetnum):
            for j in range(labelnum):
                targets[k,j] = lvalue
            for j in target_list[k]:
                targets[k,j] = hvalue
        return targets

    def target_mat2vec(self, targets, labelnum, thresh = 0.5):
        target_list = []
        if thresh > 0:
            for k in range(targets.shape[0]):
                l = []
                for j in range(targets.shape[1]):
                    if targets[k,j] >= thresh:
                        l.append(j)
                target_list.append(l)
        if thresh < -1024.0:
            for k in range(targets.shape[0]):
                l = []
                m1 = targets[k,:].max()
                for j in range(targets.shape[1]):
                    if np.abs(targets[k,j] - m1) < 0.01:
                        l.append((j,m1)) #label and confidence
                target_list.append(l) 
        else: #top value
            for k in range(targets.shape[0]):
                l = []
                m1 = targets[k,:].max()
                for j in range(targets.shape[1]):
                    if np.abs(targets[k,j] - m1) < 0.01:
                        l.append(j)
                target_list.append(l)
        return target_list 
        
    def save(self):
        if None == self._modelpath:
            return -1
        with open(self._modelpath, 'wb') as f:
            pickle.dump((self._cost, self._train, self._predict, self._minmax,self._mlp), f)
        return 0

    def load(self):
        if None == self._modelpath:
            return -1
        with open(self._modelpath, 'rb') as f:
            self._cost, self._train, self._predict, self._minmax, self._mlp = pickle.load(f)
        return 0




