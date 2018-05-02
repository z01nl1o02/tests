import os,sys,pdb,cPickle
import numpy as np
import cv2,math
import feat
import argparse 
from matplotlib import pyplot as plt

class FEAT(object):
    def __init__(self,root):
        for rdir, pdirs, names in os.walk(root):
            for name in names:
                sname,ext = os.path.splitext(name)
                if ext != '.jpg':
                    print 'unk file ext ',ext
                    continue
                img = cv2.imread(os.path.join(rdir,name),1)
                vec = feat.color.get(img)
                vec = [math.log(x + 1) for x in vec] #transform to be norm 
                with open( os.path.join(rdir,sname) + '.pkl', 'wb') as f:
                    cPickle.dump(vec, f)
        return

class GAUSSIAN_ONE(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
        return
    def calc(self, x):
        y  = (x - self.mean)**2
        y = math.exp( -y / (2*self.std**2) )
        #y = y / (self.std * math.sqrt(2*math.pi) )
        return y

class VIEW(object):
    def __init__(self, root, histsize = 5):
        data = []
        for rdir, pdirs, names in os.walk(root):
            for name in names:
                sname,ext = os.path.splitext(name)
                if ext != '.pkl':
                    continue
                if sname == 'model':
                    continue
                with open( os.path.join(rdir,sname) + '.pkl', 'rb') as f:
                    vec =cPickle.load(f)
                data.append( vec )
        data = np.asarray(data)
        plt.figure()
        N = np.maximum(9,data.shape[1])
        for k in range(data.shape[1]):
            plt.subplot('33%d'%(k+1))
            plt.hist(data[:,k],histsize)
        plt.legend()
        plt.show()
        return


class TRAIN(object):
    def __init__(self, root):
        self.models = []
        mean = []
        total = 0
        for rdir, pdirs, names in os.walk(root):
            for name in names:
                sname,ext = os.path.splitext(name)
                if ext != '.pkl':
                    continue
                with open( os.path.join(rdir,sname) + '.pkl', 'rb') as f:
                    vec =cPickle.load(f)
                if len(mean) == 0:
                    mean = vec
                else:
                    mean = (np.asarray(mean) + np.asarray(vec)).tolist()
                total += 1
        mean = [ x/total for x in mean]
        std = []
        for rdir, pdirs, names in os.walk(root):
            for name in names:
                sname,ext = os.path.splitext(name)
                if ext != '.pkl':
                    continue
                with open( os.path.join(rdir,sname) + '.pkl', 'rb') as f:
                    vec =cPickle.load(f)
                if len(std) == 0:
                    std = [ (x - m)**2 for x,m in zip(vec,mean)]
                else:
                    std = [ (x - m)**2 + s for x,m, s in zip(vec,mean,std)]
        std = [ math.sqrt(x/total) for x in std]
        for m,s in zip(mean, std):
            self.models.append( GAUSSIAN_ONE(m,s) )
        with open(os.path.join(root,'model.pkl'), 'wb') as f:
            cPickle.dump(self.models, f)
        return


class TEST(object):
    def __init__(self,root):
        logs = []
        with open(os.path.join(root, 'model.pkl'),'rb') as f:
            self.models = cPickle.load(f)
        for rdir, pdirs, names in os.walk(root):
            for name in names:
                sname,ext = os.path.splitext(name)
                if ext != '.pkl':
                    continue
                if sname == 'model':
                    continue
                with open( os.path.join(rdir,sname) + '.pkl', 'rb') as f:
                    vec =cPickle.load(f)
                Prs = [ m.calc(x) for x,m in zip(vec, self.models)]
                Pr = reduce( lambda x,y:x*y, Prs)
                print Prs
                logs.append( '%s|%lf'%(os.path.join(rdir,name),Pr))
        with open(os.path.join(root,'test.log'), 'wb') as f:
            f.writelines('\r\n'.join(logs) )
        return



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('mode',help="feat/train/test")
    ap.add_argument('root',help="input root dir")
    args = ap.parse_args()
    if args.mode == "feat":
        FEAT(args.root)
    elif args.mode == "train":
        TRAIN(args.root)
    elif args.mode == "test":
        TEST(args.root)
    elif args.mode == 'view':
        VIEW(args.root)
    return

if __name__=="__main__":
    main()
