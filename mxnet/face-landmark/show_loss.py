import os,sys,pdb
import cPickle
from matplotlib import pyplot as plt

filepath = 'models/loss.pkl'

with open(filepath,'rb') as f:
    train_loss,test_loss,train_lr = cPickle.load(f)
    
plt.figure()
X = []
Y = []
for (x,y) in train_loss:
    if x < 500:
        continue
    X.append(x)
    Y.append(y)
plt.plot(X,Y,color='red',label='train loss')


X = []
Y = []
for (x,y) in test_loss:
    X.append(x)
    Y.append(y)
plt.plot(X,Y,color='blue',label='test loss')

plt.legend()
plt.show()