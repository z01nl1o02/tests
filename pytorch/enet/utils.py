from __future__ import print_function
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
import os,copy
import cv2


class CycleScheduler:
    def __init__(self,updates_one_cycle, min_lr, max_lr):
        self.updates_one_cycle = np.float32(updates_one_cycle)
        self.min_lr = min_lr
        self.max_lr = max_lr
        return

    def __call__(self,update):
        update = update % self.updates_one_cycle
        lr = self.min_lr + (self.max_lr - self.min_lr) * update / self.updates_one_cycle
        return lr

class LinearScheduler:
    def __init__(self, update_total,min_lr, max_lr):
        self.updates_total = np.float32(update_total)
        self.min_lr = min_lr
        self.max_lr = max_lr
        return

    def __call__(self,update):
        ratio = 1 - update / self.updates_total
        lr = self.min_lr + (self.max_lr - self.min_lr) * ratio
        return lr

class Accuracy:
    def __init__(self, name="acc"):
        self.name = name
        self.correct = 0
        self.total = 0
    def reset(self):
        self.correct = 0
        self.total = 0
        return
    def get(self):
        if self.total <= 0:
            self.total = 1
        return '{}={}'.format(self.name, self.correct * 1.0 / self.total)
    def update(self,pred,label):
        preds = pred.cpu()
        labels = label.cpu()
        preds = torch.argmax(preds, dim=1).to(dtype=torch.long)
        a,b = preds.numpy(), labels.numpy()
        self.correct += torch.sum(preds == labels).item()
        self.total += labels.size(0) * labels.size(1) * labels.size(2)
        return


def show_seg_mask(ind,Y,out):
    if os.path.exists("debug"):
        Y = Y.cpu()
        out = out.cpu()
        groundtruth = Y[0].numpy() * 10
        out = out[0].numpy()
        out = np.argmax(out,axis=0) * 10
        cv2.imwrite("debug/{}_groundtruth.jpg".format(ind),np.uint8(groundtruth))
        cv2.imwrite("debug/{}_test.jpg".format(ind),np.uint8(out))

def test_seg(model,loader, device, class_weights, dtype=torch.float32, verbose=True):
    acc = Accuracy()
    ce_loss = nn.CrossEntropyLoss(weight=class_weights.to(device=device))
    model.eval()  # set model to evaluation mode
    loss_epoch = 0
    with torch.no_grad():
        for ind, (x, y) in enumerate(loader):
            x = x.to(device=device, dtype=dtype)  # move to device, e.g. GPU
            y = y.to(device=device, dtype=torch.long)
            scores = model(x)
            loss = ce_loss(scores, y)
            loss_epoch += loss.item()
            acc.update(pred=scores, label = y)
            show_seg_mask(ind,y,scores)
        loss_epoch /= len(loader.dataset)
        if verbose:
            print('\ttest loss {} {}'.format(loss_epoch,acc.get()))
    return

def update_learning_rate(optimizer, update, lr_sch):
    lr = lr_sch(update)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return

def train_seg(model, optimizer, loader_train, loader_test, device, lr_sch,class_weights=None,dtype=torch.float32, epochs=100):
    ce_loss = nn.CrossEntropyLoss(weight=class_weights.to(device=device))
    model = model.to(device=device)  # move the model parameters to CPU/GPU
    acc = Accuracy()
    update_num = 0
    for e in range(epochs):
        loss_epoch = []
        acc.reset()
        for t, (x, y) in enumerate(loader_train):
            update_num += 1
            update_learning_rate(optimizer, update_num, lr_sch)
            model.train()  # put model to training mode
            x = x.to(device=device, dtype=dtype)  # move to device, e.g. GPU
            y = y.to(device=device, dtype=torch.long)

            scores = model(x)
            loss = ce_loss(scores, y)

            # Zero out all of the gradients for the variables which the optimizer
            # will update.
            optimizer.zero_grad()

            # This is the backwards pass: compute the gradient of the loss with
            # respect to each  parameter of the model.
            loss.backward()

            # Actually update the parameters of the model using the gradients
            # computed by the backwards pass.
            optimizer.step()

            acc.update(pred = scores, label = y)
            loss_epoch.append( loss.item() )
        loss_epoch = reduce(lambda a,b: a + b, loss_epoch) / len(loader_train.dataset)
        print("epoch {} loss {} {}".format(e, loss_epoch, acc.get()))
        test_seg(model, loader_test, device, class_weights)

