import cPickle,gzip
import cv2,shutil,sys,os
import numpy as np

f = gzip.open('mnist.pkl.gz', 'rb')
train_set, valid_set, test_set = cPickle.load(f)
f.close()
os.mkdir('mnist')
os.mkdir('mnist\\train')
os.mkdir('mnist\\valid')
os.mkdir('mnist\\test')

print 'train'
for k in range(train_set[0].shape[0]):
    img = train_set[0][k,:]
    img = np.reshape(img, (28,28)) * 255
    label = train_set[1][k]
    targetfolder = 'mnist\\train\\' + str(label) + '\\'
    if False == os.path.exists(targetfolder):
        os.mkdir(targetfolder)
    cv2.imwrite(targetfolder+str(k)+'.jpg', img)
    print '.',
print ' '

print 'test'
for k in range(test_set[0].shape[0]):
    img = test_set[0][k,:]
    img = np.reshape(img, (28,28)) * 255
    label = test_set[1][k]
    targetfolder = 'mnist\\test\\' + str(label) + '\\'
    if False == os.path.exists(targetfolder):
        os.mkdir(targetfolder)
    cv2.imwrite(targetfolder+str(k)+'.jpg', img)
    print '.',
print ' '

print 'valid'
for k in range(valid_set[0].shape[0]):
    img = valid_set[0][k,:]
    img = np.reshape(img, (28,28)) * 255
    label = valid_set[1][k]
    targetfolder = 'mnist\\valid\\' + str(label) + '\\'
    if False == os.path.exists(targetfolder):
        os.mkdir(targetfolder)
    cv2.imwrite(targetfolder+str(k)+'.jpg', img)
    print '.',
print ' '


