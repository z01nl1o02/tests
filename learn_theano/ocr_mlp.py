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

def gradient_updates_momentum(cost, params, learning_rate, momentum):
    assert momentum < 1 and momentum >= 0
    updates = []
    for param in params:
        param_update = theano.shared(param.get_value() * 0., broadcastable=param.broadcastable)
        updates.append((param, param - learning_rate * param_update))
        updates.append((param_update, momentum * param_update + (1. - momentum)*T.grad(cost, param)))
    return updates

class SAMPLE:
    def extract_feature(self, imgpath):
        img = cv2.imread(imgpath, 0)
        stdw,stdh = (8,16)
        img = cv2.resize(img, (stdw,stdh))
        feat = np.zeros((1,stdw * stdh))
        k = 0
        for y in range(stdh):
            for x in range(stdw):
                v = img[y,x] / 255.
                feat[0,k] = v
                k += 1
        return feat

    def load_label_sample(self, rootdir, label):
        label_list = []
        sample_list = []
        sampledir = rootdir + str(label)
        for rdir, pdir, names in os.walk(sampledir):
            for name in names:
                sname, ext = os.path.splitext(name)
                if 0 == cmp('.jpg', ext):
                    fname = os.path.join(rdir, name)
                    feat = self.extract_feature(fname)
                    label_list.append(label)
                    sample_list.append(feat)
        return (label_list, sample_list)

    def shuffle(self,samples, targets, targets2):
        totalnum = samples.shape[0]
        idx = range(totalnum)
        random.shuffle(idx)
        new_samples = np.zeros(samples.shape)
        new_targets = np.zeros((len(targets),))
        new_targets2 = np.zeros(targets2.shape)
        for k in range(len(idx)):
            i = idx[k]
            new_samples[k,:] = samples[i,:]
            new_targets[k] = targets[i]
            new_targets2[:,k] = targets2[:,i]
        return (new_samples, new_targets, new_targets2) 


    def load_all_samples(self, rootdir):
        label_list = []
        sample_list = []
        for label in range(0,10,1):
            l,s = self.load_label_sample(rootdir, label)
            label_list.extend(l)
            sample_list.extend(s)
        sample_num = len(sample_list)
        sample_dim = sample_list[0].shape[1]
        label_num = sample_num
        label_dim = len(set(label_list))

        targets = np.array(label_list)
        samples = np.zeros( (sample_num,sample_dim ) ) 
        for k in range(len(sample_list)):
            samples[k,:] = sample_list[k]
        targets2 = np.zeros((label_dim, label_num)) #the targets compatible with net output
        for k in range(label_num):
            l = targets[k]
            targets2[l, k] = 1      
        return self.shuffle(samples,targets, targets2)

def calc_accuration(targets, outputs):
#    return np.mean((outputs > 0.5) == targets)
    hit = 0
    for k in range(outputs.shape[1]):
        o = outputs[:,k]
        t = targets[0,k]
        num = 0
        for i in o:
            if np.abs(i - o.max()) < 0.001:
                num += 1
        if num > 1:
            continue
        if np.abs(o[t] - o.max()) < 0.001:
            hit += 1 
    return hit * 1. / outputs.shape[1]
 
def train_mlp(rootdir):
    spl = SAMPLE()
    samples, targets, targets2 = spl.load_all_samples(rootdir)
    sample_num, sample_dim = samples.shape
    target_num = len(targets)
    target_dim = len(set(targets)) # -1 for binary-classes problem
    targets = np.reshape(targets, (1, -1))
    assert sample_num == target_num
    layer_sizes = [sample_dim, 1 * sample_dim, target_dim]
    W_init = []
    b_init = []
    activations = []
    for n_input, n_output in zip(layer_sizes[:-1], layer_sizes[1:]):
        W_init.append(np.random.randn(n_output, n_input))
        b_init.append(np.ones(n_output))
        activations.append(T.nnet.sigmoid)
    mlp = MLP(W_init, b_init, activations)
    mlp_input = T.matrix('mlp_input')
    mlp_target = T.matrix('mlp_target')
    learning_rate = 0.01
    momentum = 0.9
    cost = mlp.squared_error(mlp_input, mlp_target)
    train = theano.function([mlp_input,mlp_target], cost, updates=gradient_updates_momentum(cost, mlp.params, learning_rate, momentum))
    predict = theano.function([mlp_input],  mlp.output(mlp_input))
    
    iteration = 0
    max_iteration = 100
    samplesT = np.transpose(samples) #W*x + b
    while iteration < max_iteration:
        if 1:
            for k in range(samplesT.shape[1]):
                s = np.reshape(samplesT[:,k],(-1,1))
                t = np.reshape(targets2[:,k],(-1,1))
                current_cost = train(s,t)
            
        current_cost = train(samplesT,targets2)
        current_output = predict(samplesT)
        print iteration, ' ', current_cost.mean(), ' ', calc_accuration(targets, current_output)
        iteration += 1

    with open('model.ocr_mlp.txt', 'w') as f:
        pickle.dump(predict, f)

def predict_mlp(rootdir):
    spl = SAMPLE()
    samples, targets, targets2 = spl.load_all_samples(rootdir)
    with open('model.ocr_mlp.txt', 'r') as f:
        predict = pickle.load(f)

    samplesT = np.transpose(samples) #W*x + b
    output = predict(samplesT)
    targets = np.reshape(targets,(1,-1))
    print 'accuration ', calc_accuration(targets, output)

if __name__=='__main__':
    if len(sys.argv) >= 2:
        if 0 == cmp(sys.argv[1], '-train'):
            train_mlp(sys.argv[2])
        elif 0 == cmp(sys.argv[1], '-test'):
            predict_mlp(sys.argv[2])
        else:
            print 'unknown option'
    else:
        print 'lack of input'
