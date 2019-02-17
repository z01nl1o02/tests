##########################
#https://www.kaggle.com/jiqiujia/elastic-transform-for-data-augmentation
#########################

import numpy as np
import cv2
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter

# Function to distort image
def elastic_transform(image, alpha, sigma, alpha_affine, random_state=None):
    """Elastic deformation of images as described in [Simard2003]_ (with modifications).
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
         Convolutional Neural Networks applied to Visual Document Analysis", in
         Proc. of the International Conference on Document Analysis and
         Recognition, 2003.

     Based on https://gist.github.com/erniejunior/601cdf56d2b424757de5
    """
    if random_state is None:
        random_state = np.random.RandomState(None)

    shape = image.shape
    shape_size = shape[:2]

    # Random affine
    if alpha_affine > 0:
        center_square = np.float32(shape_size) // 2
        square_size = min(shape_size) // 3
        pts1 = np.float32([center_square + square_size, #point 1
                           [center_square[0] + square_size, center_square[1] - square_size], #point 2
                           center_square - square_size]) #point 3
        pts2 = pts1 + random_state.uniform(-alpha_affine, alpha_affine, size=pts1.shape).astype(np.float32)
        M = cv2.getAffineTransform(pts1, pts2)
        image = cv2.warpAffine(image, M, shape_size[::-1], borderMode=cv2.BORDER_REFLECT_101)
        image = image[...,np.newaxis]


    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    dz = np.zeros_like(dx)

    x, y, z = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), np.arange(shape[2]))

    indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1)), np.reshape(z+dz,(-1,1))

    image_t =  map_coordinates(image, indices, order=1, mode='reflect').reshape(shape)
    return image_t[...,0]

# Load images
im = cv2.imread("c:/temp/1.png", 0)


# Apply transformation on image
min_size = np.minimum( im.shape[0],im.shape[1] )
alpha = min_size * 2
beta = min_size * 0.08
alpha_affine = min_size * 0.08
#alpha_affine = -1
im_t = elastic_transform(im[...,np.newaxis], alpha,beta,alpha_affine)

vis = np.vstack((im,im_t))
cv2.imshow("result",vis)
cv2.waitKey(-1)


