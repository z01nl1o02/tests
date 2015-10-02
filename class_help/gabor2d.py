import numpy as np
import math
import pylab as plt
import pdb

def create_gabor_2d(bandwidth, x2y, psi, wavelen, theta):
#bandwidth: 1
#x2y: aspect ratio, 1
#psi: phase shift,0
#wavelen: >=2
#theta: angle in radian [0,pi]
    sigma = wavelen / math.pi * math.sqrt(math.log(2) / 2.0) * (math.pow(2,bandwidth)+1) / (math.pow(2,bandwidth) - 1)
    sigma_x = sigma
    sigma_y = sigma / x2y

    sz = np.int32(4 * np.maximum(sigma_x, sigma_y))

    if 0 == (sz % 2):
        sz += 1

    r = np.int32(sz/2)
    X = np.zeros((2*r,2*r))
    Y = np.zeros((2*r,2*r))
    for y in range(-r,r,1):
        for x in range(-r,r,1):
            X[y+r,x+r] = x
            Y[y+r,x+r] = y
    x_theta = X * math.cos(theta) + Y * math.sin(theta)
    y_theta = -X * math.sin(theta) + Y * math.cos(theta)

    gb = np.exp( -0.5 * (x_theta **2 / (sigma_x**2) + y_theta ** 2 / (sigma_y**2)))
    gb = gb * np.cos(2 * math.pi / wavelen * x_theta + psi)
    return gb


if __name__=="__main__":
    plt.figure()
    k = 1
    for wl in range(5,16,5):
        for a in range(0,180,30):
            plt.subplot(3,6,k)
            k += 1
            gb = create_gabor_2d(1,1,0,wl,math.pi*a / 180.0)
            plt.imshow(gb,cmap = plt.cm.gray)
    plt.show()   








