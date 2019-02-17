import mxnet as mx
from mxnet.gluon import Trainer
from datasets import segment_voc
from networks import enet
from utils import train_seg,CycleScheduler,LinearScheduler
import os
import numpy as np
import torch, torchvision
import torch.optim as optim


device = torch.device("cuda:0")
batch_size = 10
num_epochs = 100
base_lr = 0.001 #should be small for model with pretrained model
wd = 0.0005
net_name = "enet"
dataset_name = 'voc'
load_to_train = False
output_folder = os.path.join("output")
output_prefix = os.path.join(output_folder,net_name+"_")


if not os.path.exists(output_folder):
    os.makedirs(output_folder)

if dataset_name == 'voc':
    class_names = segment_voc.get_class_names()
    train_iter, test_iter, num_train = segment_voc.load(batch_size)

if net_name == "enet":
    net = enet.get_net(len(class_names))


net = net.cuda()

optimizer = optim.SGD(net.parameters(), lr=base_lr, momentum=0.99)

iter_per_epoch = num_train // batch_size
#lr_sch = CycleScheduler(updates_one_cycle = iter_per_epoch * 5, min_lr = base_lr/10, max_lr = base_lr * 5)
lr_sch = LinearScheduler(iter_per_epoch * num_epochs, min_lr=base_lr/1000, max_lr=base_lr)



class_weights = [10 for k in range(len(class_names))]
class_weights[0] = 0.1
class_weights = torch.from_numpy( np.asarray(class_weights,dtype=np.float32) )
train_seg(net,optimizer, train_iter, test_iter,device,lr_sch,class_weights=class_weights,epochs=num_epochs)

