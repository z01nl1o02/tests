import os,sys
import numpy as np


class CONFIG(object):
    def __init__(self):
        self.data_root = "C:/dataset/cifar/split"

        self.width, self.height = 28,28
        self.batch_size = 64
        self.base_lr = 0.01
        self.max_epoch = 100

        self.class_num = 10

        return
    @property
    def train_root(self):
        return os.path.join(self.data_root,'train')
    @property
    def test_root(self):
        return os.path.join(self.data_root,"test")

config = CONFIG()