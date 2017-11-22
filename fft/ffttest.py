import cv2,pdb
import numpy as np

def cvt2img(f):
    f = (f - f.min())  / (f.max() - f.min()) 
    a = 20
    f = np.log( a * f + 1) / np.log(a + 1)
    return np.uint8(f * 255)
def show_freq_image(win,F):
    mag = cvt2img( np.abs(F) )
    real = cvt2img( np.real(F) )
    imag = cvt2img( np.imag(F) )
    angle = cvt2img( np.angle(F) )
    canvas1 = np.hstack((mag,angle))
    canvas2 = np.hstack((real,imag))
    canvas = np.vstack( (canvas1, canvas2) )
    cv2.imshow(win,canvas)

img1 = cv2.imread('1.jpg',0)
img2 = cv2.imread('2.jpg',0)

cv2.imshow('1',img1)
cv2.imshow('2',img2)

#standard
F1 = np.fft.fft2(img1 * 1.0)
F2 = np.fft.fft2(img2 * 1.0)
show_freq_image("F1",np.fft.fftshift(F1))
show_freq_image("F2",np.fft.fftshift(F2))

#conjugate ifft
IF1conj = np.fft.ifft2(F1.conjugate()) 
IF1 = np.fft.ifft2(F1)
show_freq_image("IF1Conj",IF1conj)
show_freq_image("IF1",IF1)


IF2conj = np.fft.ifft2(F2.conjugate()) 
IF2 = np.fft.ifft2(F2)
show_freq_image("IF2Conj",IF2conj)
show_freq_image("IF2",IF2)

#exchange mag/angle
A1 = np.angle(F1)
M1 = np.abs(F1)

A2 = np.angle(F2)
M2 = np.abs(F2)

M1A2 = np.zeros(img1.shape, dtype=np.complex64)
M1A2.real = M1 * np.cos(A2)
M1A2.imag = M1 * np.sin(A2)
iM1A2 = np.fft.ifft2(M1A2)
show_freq_image("M1-A2",iM1A2)

M2A1 = np.zeros(img1.shape, dtype=np.complex64)
M2A1.real = M2 * np.cos(A1)
M2A1.imag = M2 * np.sin(A1)
iM2A1 = np.fft.ifft2(M2A1)
show_freq_image("M2-A1",iM2A1)




cv2.waitKey()
