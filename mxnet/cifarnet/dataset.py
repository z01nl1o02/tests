import numpy as np
import os,sys,cv2
import mxnet as mx
from mxnet import gluon
from config import config
from collections import defaultdict
from copy import deepcopy
import random
import logging

class DS_CIFAR(gluon.data.Dataset):
    def __init__(self,root, fortrain, crop_hw):
        self.pair_list = []
        self.fortrain = fortrain
        self.crop_hw = crop_hw
        self.center_crop = False
        self.label2id = defaultdict(int)
        with open(os.path.join(root,'../labels.txt'), 'rb') as f:
            for idx, line in enumerate(f):
                line = line.strip()
                if line == "":
                    continue
                self.label2id[line] = idx
                logging.info("label {} label ID {}".format(line,idx))
        def get_path_label_pair(filepath):
            pairs = []
            with open(filepath,'rb') as f:
                for line in f:
                    line = line.strip()
                    if line == "":
                        continue
                    path = line
                    label = self.label2id[line.split('\\')[-2]]
                    pairs.append((path,label))
            return pairs
        if fortrain:
            filepath = os.path.join(root,'../train.txt')
        else:
            filepath = os.path.join(root,'../test.txt')
        self.pair_list.extend(get_path_label_pair(filepath))
        return
    def __len__(self):
        return len(self.pair_list)

    def _augment_crop(self, img, crop_hw, center_crop=False):
        H,W,C = img.shape
        if self.fortrain == False or center_crop == True:
            dx = (W - crop_hw[1])//2
            dy = (H - crop_hw[0])//2
        else:
            dx = random.randint(0, (W - crop_hw[1])//2)
            dy = random.randint(0, (H - crop_hw[0])//2)
        return img[dy:dy+crop_hw[0], dx:dx+crop_hw[1],:]
    def _augment_flip(self, img):
        return cv2.flip(img,1)

    def _load_pair(self,idx):
        path,label = self.pair_list[idx]
        img = cv2.imread(path,1)

        if self.fortrain == True and self.center_crop == False:
            img = self._augment_crop(img,self.crop_hw)
        else:
            img = self._augment_crop(img,self.crop_hw, True)

        if self.fortrain == True and random.randint(0,100) > 50:
            img = self._augment_flip(img)


        img = np.transpose(np.float32(img),(2,0,1))
        label = np.asarray([label])
       # print img.shape



        return (img,label)
    def __getitem__(self, idx):
        return self._load_pair(idx)




class DS_MNIST(gluon.data.Dataset):
    def __init__(self,root, fortrain, crop_hw):
        self.pair_list = []
        self.fortrain = fortrain
        self.crop_hw = crop_hw
        self.center_crop = False
        self.label2id = defaultdict(int)
        with open(os.path.join(root,'../labels.txt'), 'rb') as f:
            for idx, line in enumerate(f):
                line = line.strip()
                if line == "":
                    continue
                self.label2id[line] = idx
                logging.info("label {} label ID {}".format(line,idx))
        def get_path_label_pair(filepath):
            pairs = []
            with open(filepath,'rb') as f:
                for line in f:
                    line = line.strip()
                    if line == "":
                        continue
                    path = line
                    label = self.label2id[line.split('\\')[-2]]
                    pairs.append((path,label))
            return pairs
        if fortrain:
            filepath = os.path.join(root,'../train.txt')
        else:
            filepath = os.path.join(root,'../test.txt')
        self.pair_list.extend(get_path_label_pair(filepath))
        return
    def __len__(self):
        return len(self.pair_list)

    def _augment_crop(self, img, crop_hw, center_crop=False):
        H,W = img.shape
        if self.fortrain == False or center_crop == True:
            dx = (W - crop_hw[1])//2
            dy = (H - crop_hw[0])//2
        else:
            dx = random.randint(0, (W - crop_hw[1])//2)
            dy = random.randint(0, (H - crop_hw[0])//2)
        return img[dy:dy+crop_hw[0], dx:dx+crop_hw[1]]

    def _augment_noise(self,img):
        H,W = img.shape
        noise = np.asarray([random.gauss(mu = 0,sigma=50) for k in range(H*W)])
        noise = np.reshape(noise,(H,W))
        blur = np.float32(img) + noise
        blur = np.maximum(0,blur)
        blur = np.minimum(255,blur)
        blur = np.uint8(blur)
        #cv2.imshow("noise",blur)
        #cv2.waitKey(-1)
        return blur

    def _augment_blur(self,img):
        blur = cv2.blur(img,(3,3))
        return blur

    def _load_pair(self,idx):
        path,label = self.pair_list[idx]
        img = cv2.imread(path,0)

        if self.fortrain == True and self.center_crop == False:
            img = self._augment_crop(img,self.crop_hw)
        else:
            img = self._augment_crop(img,self.crop_hw, True)

        if self.fortrain == True and random.randint(0,100) > 50:
            img = self._augment_noise(img)

        if self.fortrain == True and random.randint(0,100) > 50:
            img = self._augment_blur(img)

        img = np.expand_dims(img,0)
        img = np.tile(img,(3,1,1))
        img = np.float32(img)
        label = np.asarray([label])
        return (img,label)

    def __getitem__(self, idx):
        return self._load_pair(idx)


