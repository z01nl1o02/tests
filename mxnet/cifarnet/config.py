import os,sys
import numpy as np


class CONFIG(object):
    def __init__(self):
        self.dataset = "mnist"
        if self.dataset == "cifar":
            self.data_root = os.path.join( os.getcwd(), 'data' )
        elif self.dataset == "mnist":
            self.data_root = "c:/dataset/mnist/"


        log_dir = os.path.join(os.getcwd(),'log')
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        self.log = os.path.join( log_dir, "train_{}.log".format(self.dataset))

        model_dir = os.path.join(os.getcwd(), "models_{}".format(self.dataset))
        if not os.path.exists(model_dir):
            os.makedirs( model_dir )
        self.model_prefix = os.path.join(model_dir, "cifarnet_epoch_")


        if self.dataset == "cifar":
            self.width, self.height = 64,64
            self.batch_size = 64
            self.base_lr = 0.001 #0.001
            self.max_epoch = 2000
            self.save_epoch_step = self.max_epoch // 10
            self.class_num = 2
        elif self.dataset == "mnist":
            self.width, self.height = 24,24
            self.batch_size = 100
            self.base_lr = 0.001 #0.001
            self.max_epoch = 200
            self.save_epoch_step = self.max_epoch // 100
            self.class_num = 10

        self.optimizer = "adam"

        self.pretrained_model = None #'models/used/cifarnet_epoch_{:0>5d}.params'.format(200)

        return
    @property
    def train_root(self):
        return os.path.join(self.data_root,'train')
    @property
    def test_root(self):
        return os.path.join(self.data_root,"test")

    def model_path(self,epoch):
        if isinstance(epoch,str):
            return "{}{}.params".format(self.model_prefix,epoch)
        return "{}{:0>5d}.params".format(self.model_prefix,epoch)

config = CONFIG()