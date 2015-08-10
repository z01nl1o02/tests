import os,sys,pdb,pickle
import cv2
import numpy as np
import theano
import theano.tensor as T
import pylab as plt


#using logistic regression to do binary classification on plate color (blue or yellow)
#small-scale tests show good performance
class NPR_COLOR:
    def load_sample(self, filepath):
        bgr = cv2.imread(filepath,1)
        img = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        hist = np.zeros((1,90))
        for y in range(img.shape[0]):
            for x in range(img.shape[1]):
                h = img[y,x,0] / 2
                hist[0,h] += 1
        #hist = hist * 1.0 / (img.shape[0] * img.shape[1])
        hist = hist * 1.0 / hist.max()
       # for k in range(1,hist.shape[1]):
       #     hist[0,k] = hist[0,k-1] + hist[0,k]
        return hist
    def load_samples(self, rootdir):
        sample_list = []
        for rdir, pdir, names in os.walk(rootdir):
            for name in names:
                sname,ext = os.path.splitext(name)
                if  0 == cmp(ext, '.jpg'):
                    fname = os.path.join(rdir, name)
                    hist = self.load_sample(fname)
                    sample_list.append(hist)
        samples = np.zeros((len(sample_list), sample_list[0].shape[1]))
        for k in range(len(sample_list)):
            samples[k,:] = sample_list[k]
        return samples
    def train(self, posdir, negdir):
        s0 = self.load_samples(negdir)
        l0 = [0 for k in range(s0.shape[0])]
        s1 = self.load_samples(posdir)
        l1 = [1 for k in range(s1.shape[0])]

        if 0:
            plt.figure(1)
            axpos = plt.subplot(211)
            axneg = plt.subplot(212)
            pos = s1[10,:]
            neg = s0[-10,:]
            plt.sca(axpos)
            #plt.hist(pos)
            plt.plot(range(0,pos.shape[0]), pos)
            plt.sca(axneg)
            #plt.hist(neg)
            plt.plot(range(0,pos.shape[0]), neg)
            plt.show()
            pdb.set_trace()

        s = np.vstack((s0,s1))
        l0.extend(l1)
        l = np.asarray(l0)
        rng = np.random
        x = T.matrix('x')
        y = T.vector('y')
        w = theano.shared(rng.randn(s.shape[1]),name='w')
        b = theano.shared(0., name='b')
        print 'initilal model done'
        p_1 = 1 / (1 + T.exp(-T.dot(x,w)-b))
        pred = p_1 > 0.5
        xent = -y * T.log(p_1) - (1-y) * T.log(1-p_1) #cross-entropy loss function
        cost = xent.mean() + 0.01 * (w ** 2).sum() #cost to minimize
        gw,gb = T.grad(cost, [w,b])

        trainK = theano.function(inputs=[x,y], outputs=[pred,xent], updates=((w, w-0.1*gw), (b, b-0.1*gb)))
        predictK = theano.function(inputs=[x], outputs=pred)

        with open('npr_color_trainK.func.txt', 'w') as f:
            pickle.dump(trainK, f)

        for k in range(400):
            pred, err = trainK(s, l)
            print k, ' ', err.mean(), ' ', (pred == l).sum()

        with open('npr_color_predictK.func.txt', 'w') as f:
            pickle.dump(predictK, f) #save model after training the train parameter will be saved too
        print w.get_value()
        print b.get_value()

    def predict(self,testdir):
        with open('npr_color_predictK.func.txt', 'r') as f:
            predictK = pickle.load(f)
        s = self.load_samples(testdir)
        pred = predictK(s)
        """
        for i in pred:
            print i,
        print ' '
        """
        return pred.sum()

if __name__=='__main__':
    engine = NPR_COLOR()
    engine.train('color\\train\\blue\\', 'color\\train\\yellow\\')
    s = engine.predict('color\\train\\blue\\')
    print 'blue train set ', s
    s = engine.predict('color\\test\\blue\\')
    print 'blue test set ', s
    s = engine.predict('color\\train\\yellow\\')
    print 'yellow train set ', s
    s = engine.predict('color\\test\\yellow\\')
    print 'yellow test set ', s
