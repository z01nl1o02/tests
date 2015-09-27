import os,sys,pdb,cv2,pickle,random 
import numpy as np
import theano
import theano.tensor as T

class PARAM(object):
    def __init__(self):
        self.imgw = 16
        self.imgh = 32

class LAYER(object):
    def __init__(self, init_W, init_b, tranfunc):
        output_num, input_num = init_W.shape
        assert (output_num,) == init_b.shape
        self.W = theano.shared(value=init_W.astype(theano.config.floatX), name="W")
        self.b = theano.shared(value=init_b.reshape(output_num,1).astype(theano.config.floatX), name="b", broadcastable=(False,True))
        self.tranfunc = tranfunc
        self.param = [self.W, self.b]

    def output(self,x):
        lin_out = T.dot(self.W,x) + self.b
        if self.tranfunc is None:
            return lin_out
        return self.tranfunc(lin_out)


class MLP(object):
    def __init__(self, init_Ws, init_bs, tranfuncs):
        self.layers = []
        self.params = []

        for init_W, init_b, tranfunc in zip(init_Ws, init_bs, tranfuncs):
            layer = LAYER(init_W, init_b, tranfunc)
            self.layers.append(layer)
            self.params.extend(layer.param)


    def output(self,x):
        for layer in self.layers:
            x = layer.output(x)
        return x

    def calc_error(self,x,y):
        py = self.output(x)
        error = T.sum((py - y) ** 2)
        wc = 0
        for param in self.params:
            wc += np.sum(param.get_value() ** 2)
        error =error * 0.5 + wc * 0.5
        return error

    def gen_update_pairs(self, cost, learn_ratio, moment):
        updates = []
        for param in self.params:
            upd = theano.shared(param.get_value() * 0., broadcastable=param.broadcastable)
            updates.append( (param, param - learn_ratio * upd   )   )
            updates.append( (upd, moment * upd + (1.0 - moment) * T.grad(cost, param) ) )
        return updates 


def gen_initial_param(layer_sizes):
    assert 3 == len(layer_sizes)
    Ws = []
    bs = []
    tfs = []
    for in_num, out_num in zip(layer_sizes[:-1], layer_sizes[1:]):
        W = np.random.randn(out_num, in_num)
        b = np.ones(out_num)
        tf = T.nnet.sigmoid
        Ws.append(W)
        bs.append(b)
        tfs.append(tf)
    return (Ws, bs, tfs)

def create_config_net(layer_sizes):
    learn_ratio = 0.01
    moment = 0.9
    net_in = T.matrix('net_in')
    net_out = T.matrix('net_out')
    
    init_Ws, init_bs, init_tfs = gen_initial_param(layer_sizes)
    net = MLP(init_Ws, init_bs, init_tfs)

    cost = net.calc_error(net_in, net_out)
#    train = theano.function([net_in, net_out], cost, mode='DebugMode', updates = net.gen_update_pairs(cost, learn_ratio, moment))
    train = theano.function([net_in, net_out], cost, updates = net.gen_update_pairs(cost, learn_ratio, moment))
    out = net.output(net_in)
    predict = theano.function([net_in],out)

    return (net, train, predict)


class DATA(object):
    def gen_feature(self, imgpath):
        img = cv2.imread(imgpath, 0)
 
        stdw,stdh = (PARAM().imgw,PARAM().imgh)
        img = cv2.resize(img, (stdw,stdh))
        img = np.reshape(img, (1,-1))
        img = img / 255.0
        return img
    
    def load_label_sample(self, rootdir, label):
        labels = []
        samples = []
        for rdir,pdir,names in os.walk(rootdir+str(label)):
            for name in names:
                sname,ext = os.path.splitext(name)
                if 0 == cmp('.bmp', ext) or 0 == cmp('.jpg', ext):
                    sample = self.gen_feature( os.path.join(rdir,name) )
                    samples.append(sample)
                    labels.append(label)
        return (samples,labels)

    def shuffle(self, samples, labels):
        total = len(samples)
        idxs = range(total)
        random.shuffle(idxs)
        newsamples = []
        newlabels = []
        for old in idxs:
            newsamples.append(samples[old])
            newlabels.append(labels[old])
        return (newsamples, newlabels)
    
    def load(self, rootdir):
        labels = []
        samples = []
        labellist = range(0,5,1)
        for label in labellist:
            s,l = self.load_label_sample(rootdir, label)
            samples.extend(s)
            labels.extend(l)
        samples,labels = self.shuffle(samples,labels)
        return (samples, labels, labellist)

def whiten_data(samples):

    m = np.reshape(np.mean(samples,1),(-1,1))
    x = samples - np.tile(m,(1, samples.shape[1]))
    xx = np.dot(x, np.transpose(x)) / samples.shape[1]
    u,s,v = np.linalg.svd(xx)
    xRot = np.dot(np.transpose(u) , x)
    s = np.reshape(np.sqrt(s),(-1,1)) + 0.000001
    x_pcaw = xRot / np.tile(s, (1, xRot.shape[1]))
    x_zcaw = np.dot(u , x_pcaw)
    samples = x_zcaw

    return samples

def gen_train_samples(samples, labels, labellist):
    total = len(samples)
    dim = samples[0].shape[1]
    trainset = np.zeros((dim,total))
    trainlabel = np.zeros((len(labellist), total))
    for k in range(total):
        trainset[:,k] = np.transpose(samples[k]).reshape(trainset[:,k].shape)
#        trainlabel[labels[k],k] = 1

#    trainset = whiten_data(trainset)
    trainlabel = trainset #self-map

    trainset = whiten_data(trainset)

    return (trainset, trainlabel)

def calc_false_alarm(target,output):
    hit = 0
    line = ""
    for k in range(target.shape[1]):
        tgt = target[:,k]
        prd = output[:,k]
        prdmax = prd.max()
        line += str(np.argmax(tgt)) + ' ' + str(np.argmax(prd)) + '\n'
        if prdmax < 0.001:
            continue
        maxnum = 0
        for i in prd:
            if np.abs(i - prdmax ) < 0.001:
                maxnum += 1
        if maxnum > 1:
            continue
        if np.argmax(tgt) == np.argmax(prd):
            hit += 1
    with open('net.log', 'w') as f:
        f.writelines(line)
    return 1 - hit * 1. / target.shape[1]

def draw_prd(prd):
    for k in range(5):
        img = np.zeros((PARAM().imgh,PARAM().imgw))
        for y in range(PARAM().imgh):
            for x in range(PARAM().imgw):
                img[y,x] = prd[y * PARAM().imgw + x, k]
        img = (img - img.min()) * 255.0 / (img.max() - img.min())
        img = np.uint8(img)
        cv2.imwrite('d:/tmp/ALPR/prd'+str(k)+'.jpg', img) 


def draw_weight(net):
    W = net.layers[0].W.get_value()
    for k in range(W.shape[0]):
        img = np.reshape(W[k,:],(PARAM().imgh,PARAM().imgw))
   #     norm = np.sqrt(np.sum(img ** 2))
   #     img = img / norm
        if k < 3:
            print ' ', img.min(), ' ', img.max(), ' ', W.shape
#        img = np.abs(img)
        img = (img - img.min()) * 255.0 / (img.max() - img.min())
        img = np.uint8(img)
        cv2.imwrite('d:/tmp/ALPR/ocr'+str(k)+'.jpg', img) 


def do_train(rootdir):
    samples, labels, labellist = DATA().load(rootdir)  
    print 'sample num ', len(samples)
    in_num = samples[0].shape[1]
#    layer_size = [in_num, 20, len(labellist)]
    layer_size = [in_num, 10, in_num]
    net,train,predict = create_config_net(layer_size)


    trainset,trainlabel = gen_train_samples(samples, labels, labellist)

    iterator = 0
    batchsize = 5

#    draw_weight(net)
    while iterator < 50000:
        iterator += 1
        all_cost = 0
        all_num = 0 
        for k in range(0, trainset.shape[1], batchsize):
            kk = k
            if  kk + batchsize > trainset.shape[1]:
                kk = trainset.shape[1] - batchsize
            batchidx = range(kk,kk+batchsize,1)
            cur_cost = train( trainset[:,batchidx], trainlabel[:,batchidx])
            all_cost += cur_cost.sum()
            all_num += batchsize
            
        if iterator % 5 == 0:
            prd = predict(trainset) 
            fa = calc_false_alarm(trainlabel, prd)
            print iterator, ' ', all_cost*1./all_num, ' ', fa
            draw_weight(net)
            draw_prd(prd)

if __name__=="__main__":
    if len(sys.argv) == 3 and 0 == cmp(sys.argv[1], '-train'):
        do_train(sys.argv[2])





