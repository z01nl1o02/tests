import numpy as np
import os,sys,cv2
import mxnet as mx
from mxnet import gluon
from config import config
from collections import defaultdict
from copy import deepcopy
import random
import logging

class JSDataset(gluon.data.Dataset):
    def __init__(self,root, fortrain, crop_hw, max_sample_num = -1):
        self.pair_list = []
        self.fortrain = fortrain
        self.crop_hw = crop_hw
        self.center_crop = False
        def get_path_label_pair(filepath):
            pairs = []
            with open(filepath,'rb') as f:
                for line in f:
                    line = line.strip()
                    if line == "":
                        continue
                    path = line.split(' ')[0]
                    label = np.int64( line.split(' ')[1] )
                    pairs.append((path,label))
            return pairs
        if fortrain:
            filepath = os.path.join(root,'train.txt')
        else:
            filepath = os.path.join(root,'test.txt')
        self.pair_list.extend(get_path_label_pair(filepath))

        if max_sample_num > 0 and len(self.pair_list) > max_sample_num:
            logging.info("downsampling to {}".format(max_sample_num))
            half_sample_num = max_sample_num // 2
            pos_list = filter(lambda x: x[1] == 1, self.pair_list)
            neg_list = filter(lambda x: x[1] == 0, self.pair_list)
            if len(pos_list) > half_sample_num:
                random.shuffle(pos_list)
                pos_list = pos_list[0:half_sample_num]
            if len(neg_list) > half_sample_num:
                random.shuffle(neg_list)
                neg_list = neg_list[0:half_sample_num]
            self.pair_list = pos_list
            self.pair_list.extend(neg_list)

        return
    def __len__(self):
        return len(self.pair_list)

    def _augment_crop(self, img, crop_hw, center_crop=False):
        H,W,C = img.shape
        if  center_crop == True:
            dx = (W - crop_hw[1])//2
            dy = (H - crop_hw[0])//2
        else:
            dx = random.randint(0, (W - crop_hw[1])//2)
            dy = random.randint(0, (H - crop_hw[0])//2)
        return img[dy:dy+crop_hw[0], dx:dx+crop_hw[1],:]
    def _augment_lighting(self,img):
        pwr = random.randint(6,12) / 12.0
        img = np.power(np.float64(img)/255.0, pwr)
        img = img * 255
        img = np.clip(img,0,255)
        img = np.uint8(img)
        return img
    def _augment_flip(self, img):
        return cv2.flip(img,1)
    def _augment_blur(self,img):
        rnd = random.randint(0,100)
        if rnd % 3 == 0:
            return cv2.GaussianBlur(img,(3,3),0.5)
        if rnd % 3 == 1:
            return cv2.medianBlur(img,3)
        return cv2.blur(img,(3,3))

    def _augment_rotation(self,img):
        angle = (random.randint(0,200) - 100)/10.0
        H,W,C = img.shape
        mat = cv2.getRotationMatrix2D((W/2, H/2), angle, 1)
        img = cv2.warpAffine(img,mat,(W,H))
        return img

    def _load_pair(self,idx):
        path,label = self.pair_list[idx]
        img = cv2.imread(path,1)

        if self.fortrain:
            img = self._augment_crop(img,self.crop_hw, self.center_crop)
        else:
            img = self._augment_crop(img,self.crop_hw, center_crop=True)

        if 0 and self.fortrain == True and random.randint(0,100) < 80:
            if random.randint(0,100) > 50:
                img = self._augment_flip(img)

            if random.randint(0,100) > 50:
                img = self._augment_blur(img)

            if random.randint(0,100) > 50:
                img = self._augment_lighting(img)

            if random.randint(0,100) > 50:
                img = self._augment_rotation(img)

        img = np.transpose(np.float32(img),(2,0,1)) / 255.0
        label = np.asarray([label])
        return (img,label)

    def __getitem__(self, idx):
        return self._load_pair(idx)




