import os,sys,pdb,pickle,cv2
import numpy as np
from ctypes import *

clib = cdll.LoadLibrary('ttd.dll')
ctest = clib.test
#ctest.restype = c_char_p
ctest.restype = POINTER(c_ubyte)
cfree = clib.test_free

img = cv2.imread('1.jpg', 0)

w = img.shape[1]
h = img.shape[0]

size = w * h
inptr = (c_ubyte * size)()
#inptr = (c_ubyte * w * h)() //error
for y in range(img.shape[0]):
    for x in range(img.shape[1]):
        inptr[y*img.shape[1] + x] = img[y,x]

outw = c_int(0)
outh = c_int(0)
outptr = ctest(inptr, w, h, w, pointer(outw), pointer(outh))
outw = outw.value
outh = outh.value
outsize = outw * outh
print 'in size: ', w, 'x', h
print 'out size: ', outw, 'x', outh
print 'out ptr type: ', outptr
#outptr = cast(outptr, (c_ubyte * outsize))
img = np.zeros((outh,outw))
for y in range(outh):
    for x in range(outw):
        v = outptr[y * outw + x]
        img[y,x] = v
img = np.uint8(img)
cv2.imwrite('x.jpg',img)
cfree(outptr)



