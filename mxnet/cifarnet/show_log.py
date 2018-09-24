import os,sys,re
from matplotlib import pyplot as plt
import numpy as np
from config import config
from scipy.interpolate import spline,interp1d

log_path = config.log

class CURVE(object):
    def __init__(self,name,color, smooth=True):
        self.name = name
        self.smooth = smooth
        self.color = color
        self.update_list = []
        self.value_list = []
        return
    def update(self, update, value):
        if isinstance(update,str):
            update = int(update)
        if isinstance(value,str):
            value = float(value)
        self.update_list.append(update)
        self.value_list.append(value)
        return
    def show(self):
        if self.smooth:
            X = np.asarray(self.update_list)
            Y = np.asarray(self.value_list)
            Xnew = np.linspace(X.min(), X.max(), 100)
            func = interp1d(X, Y, kind="slinear")
            Ynew = func(Xnew)
            self.update_list, self.value_list = Xnew, Ynew
        plt.plot(self.update_list, self.value_list,label=self.name, color=self.color)
        return

#train_update = CURVE(name='update',color=np.random.random((1,3)).tolist()[0])
train_lr = CURVE(name='lr',color=(0,0,0))
train_loss = CURVE(name='train loss',color=(1,0,0))
train_acc = CURVE(name='train acc',color=(1,1,0))


#test_update = CURVE(name='update',color=np.random.random((1,3)).tolist()[0])
test_loss = CURVE(name='test loss',color=(0,1,0))
test_acc = CURVE(name='test acc',color=(0,1,1))

test_acc_pos = CURVE(name='test acc pos',color=(0,0.5,0))
test_acc_neg = CURVE(name='test acc neg',color=(0.5,0,0))

with open(log_path,'rb') as f:
    for line in f:
        train = re.search(r'train update',line)
        test = re.search(r'test update',line)
        if (train is None) and (test is None):
            continue
        print line
        update = re.findall(r'update (\S+) ',line)[0]
        loss = re.findall(r'loss (\S+) ',line)[0]
        acc = re.findall(r'acc (\S+) ',line)[0]
        if not (test is None):
            test_acc.update(update,acc)
            test_loss.update(update,loss)
            acc_pos_neg = re.findall(r'acc_per_class (\S+) (\S+)', line)[0]
            test_acc_pos.update(update,acc_pos_neg[0])
            test_acc_neg.update(update,acc_pos_neg[1])
        elif not(train is None):
            lr = re.findall(r'lr (\S+) ',line)[0]
            train_acc.update(update,acc)
            train_loss.update(update,loss)
            train_lr.update(update,lr)

fig = plt.figure()
plt.ylim((0,1.2))
train_loss.show()
test_loss.show()
train_lr.show()
train_acc.show()
test_acc.show()
test_acc_pos.show()
test_acc_neg.show()
plt.legend()
fig.savefig(os.path.splitext(log_path)[0] + "_loss.jpg")
plt.show()



train_loss, train_acc = [], []
test_loss, test_acc = [], []
