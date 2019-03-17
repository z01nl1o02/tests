import os,sys,pdb
import numpy as np
import cv2
import h5py



norm_size = (64,256)
root = '/home/data/plates/VOC2018/JPEGImages/'
output_hdf5 = 'data.hdf5'

labeldict_Chi = dict() 
labelmap_Chi = open("map/chinese", "r") 
chineses = labelmap_Chi.readline().strip().split(',')
for ind,ch in enumerate(chineses): 
    labeldict_Chi[ch] = ind 
labelmap_Chi.close() 

labeldict_Letter = dict() 
labelmap_Letter = open("map/letter", "r") 
letters = labelmap_Letter.readline().strip().split(',')
for ind,letter in enumerate(letters):
    labeldict_Letter[letter] = ind
labelmap_Letter.close() 

labeldict_NL = dict() 
labelmap_NL = open("map/digitletter", "r") 
chars = labelmap_NL.readline().strip().split(',')
for ind,char in enumerate(chars):
    labeldict_NL[char] = ind
labelmap_NL.close() 

## calc mean ()
def mean(images):
    return 0
    #images = images.astype(np.float32)
    #meanValue = sum(images) / len(images)
    #return meanValue


datas = []
labels = []
for jpg in os.listdir(root):
    _,plate = jpg.split('_')
    img = cv2.imread(os.path.join(root,jpg),1)
    img = cv2.resize(img, (norm_size[1], norm_size[0]))
    img = np.transpose(img,(2,0,1)).astype(np.float32)
    label = []
    char = plate[0:3]
    if char not in labeldict_Chi:
        print 'skip ',jpg
        continue
    label.append(labeldict_Chi[char]) 
    char = plate[3]
    label.append(labeldict_Letter[char])
    
    endchar = plate[-7:-4]
    if endchar not in labeldict_NL:
        for char in plate[4:-4]:
            label.append(labeldict_NL[char])
    else:
        for char in plate[4:-7]:
            label.append(labeldict_NL[char])
        label.append(labeldict_NL[endchar])

    datas.append(img)
    label = np.asarray(label).astype(np.float32)
    label = np.reshape(label,(1,-1))
    if label.shape[1] != 7:
        print 'error shape ', label.shape, ',', jpg
    labels.append(label)

meanVal = mean(img)

datas = np.asarray(datas).astype(np.float32)
labels = np.vstack(labels)
print datas.shape
print labels.shape
with h5py.File(output_hdf5,"w") as f:
    f['data'] = datas
    f['label'] = labels


