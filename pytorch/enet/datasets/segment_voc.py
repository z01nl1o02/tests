from __future__ import print_function
import torch
import numpy as np
import torchvision
import random
import os,copy,cv2
import torch.utils.data as tudata
from PIL import Image


def pascal_palette(): #RGB mode
  palette = {(  0,   0,   0) : 0 ,
             (128,   0,   0) : 1 ,
             (  0, 128,   0) : 2 ,
             (128, 128,   0) : 3 ,
             (  0,   0, 128) : 4 ,
             (128,   0, 128) : 5 ,
             (  0, 128, 128) : 6 ,
             (128, 128, 128) : 7 ,
             ( 64,   0,   0) : 8 ,
             (192,   0,   0) : 9 ,
             ( 64, 128,   0) : 10,
             (192, 128,   0) : 11,
             ( 64,   0, 128) : 12,
             (192,   0, 128) : 13,
             ( 64, 128, 128) : 14,
             (192, 128, 128) : 15,
             (  0,  64,   0) : 16,
             (128,  64,   0) : 17,
             (  0, 192,   0) : 18,
             (128, 192,   0) : 19,
             (  0,  64, 128) : 20 }

  return palette

def convert_from_color_segmentation(label):
    arr_3d = np.asarray(label)
    #cv2.imshow("label2",arr_3d)
    #cv2.waitKey(-1)
    arr_2d = np.zeros((arr_3d.shape[0], arr_3d.shape[1]), dtype=np.uint8)
    palette = pascal_palette()
    for c, i in palette.items():
        m = np.all(arr_3d == np.array(c).reshape(1, 1, 3), axis=2)
        arr_2d[m] = i
    return arr_2d


class DatasetVOC(tudata.Dataset):
    def __init__(self,voc_sdk_root,fortrain, len_resize = 512//4, hw_crop = (512//4,512//4)):
        super(DatasetVOC,self).__init__()
        self.data_pairs = []
        self.fortrain = fortrain
        self.len_resize = len_resize
        self.hw_crop = hw_crop
        if fortrain:
            list_file = "ImageSets/Segmentation/trainval.txt"
        else:
            list_file = "ImageSets/Segmentation/val.txt"
        with open(os.path.join(voc_sdk_root,list_file),'rb') as f:
            for line in f:
                line = line.strip()
                if line == "":
                    continue
                #if len(self.data_pairs) >= 100:
                #    break
                image_path = os.path.join(voc_sdk_root,"JPEGImages/{}.jpg".format(line))
                label_path = os.path.join(voc_sdk_root,"SegmentationClass/{}.png".format(line))
                self.data_pairs.append((image_path,label_path))

        self.input_transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
        return
    def __len__(self):
        return len(self.data_pairs)

    def __getitem__(self,idx):
        image = Image.open(self.data_pairs[idx][0]).convert('RGB')
        label = Image.open(self.data_pairs[idx][1]).convert('RGB')

        W,H = image.size
        if H < W:
            h = self.len_resize
            w = h * W / H
        else:
            w = self.len_resize
            h = w * H / W

        image = image.resize((w,h),Image.BILINEAR)
        label = label.resize((w,h),Image.NEAREST)

        h,w = self.hw_crop
        W,H = image.size

        if not self.fortrain:
            image = image.crop([0,0,w,h])
            label = label.crop([0,0,w,h])
        else:
            #random crop
            #dx = random.randint(0,W - w)
            #dy = random.randint(0,H - h)
            #image = image.crop([dx,dy, dx + w, dy + h])
            #label = label.crop([dx,dy, dx + w, dy + h])
            image = image.crop([0,0,w,h])
            label = label.crop([0,0,w,h])

            #flip
	    if random.randint(0,100) > 50:
		image = image.transpose(Image.FLIP_LEFT_RIGHT)
		label = label.transpose(Image.FLIP_LEFT_RIGHT)


        image = np.asarray(image) / 255.0
        image = np.transpose(image,(2,0,1))
        label = convert_from_color_segmentation(label)
        image = torch.from_numpy(image)
        label = torch.from_numpy(label)

        return (image,label)


def load(batch_size):
    dataset_train = DatasetVOC(voc_sdk_root=os.path.join(os.environ['HOME'],"data/VOCdevkit/VOC2007/"),fortrain=True)
    dataset_test = DatasetVOC(voc_sdk_root=os.path.join(os.environ['HOME'],"data/VOCdevkit/VOC2007/"),fortrain=False)
    loader_train = tudata.DataLoader(dataset_train, batch_size = batch_size,shuffle=True, num_workers=1)
    loader_test = tudata.DataLoader(dataset_test, batch_size = batch_size,shuffle=False, num_workers=1)
    return loader_train, loader_test, len(dataset_train)

def get_class_names():
    return [k for k in range(21)]

if 0:
    loader_train, load_test, total = load(2)
    for t,(x,y) in enumerate(loader_train):
        label = np.asarray(y) * 10
        label = np.uint8(label[0])
        print('iter_in_epoch {} x_shape = {} y_shape = {}'.format(t,x.shape, y.shape))
        cv2.imshow("label",label)
        cv2.waitKey(-1)

