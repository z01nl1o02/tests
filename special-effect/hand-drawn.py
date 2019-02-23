"""
origin from
https://mp.weixin.qq.com/s?__biz=MzI4MDYzNzg4Mw==&mid=2247488926&idx=2&sn=251f18f8c9e7d511b37e567aebecffdd&chksm=ebb42d4adcc3a45c2c1e7c4a5f5b1f56ba55ee42cdb88b94804539b983e222a52790ef01ea32&scene=90&subscene=93&sessionid=1550449926&clicktime=1550450059&ascene=56&devicetype=android-27&version=27000338&nettype=cmnet&abtest_cookie=BAABAAoACwATABQABAAjlx4AWpkeAJuZHgCdmR4AAAA%3D&lang=zh_CN&pass_ticket=tBVzYNnAqHwzt3BVQdWGBW%2F6ZSATF8cFZ%2FKuCZhC6ztoYNeAyLmlzYuPO3o2CsK7&wx_header=1
"""
from PIL import Image
import numpy as np
import os,re
import time

def image(source_image,result_image,depths=10,degree_lookat = np.pi / 2.2, degree = np.pi/4):

    a = np.asarray(Image.open(source_image).convert('L')).astype('float')
    depth = depths  # depth in [0,100]
    grad = np.gradient(a)
    grad_x, grad_y = grad
    grad_x = grad_x * depth / 100.# normalization
    grad_y = grad_y * depth / 100.# normalization
    A = np.sqrt(grad_x ** 2 + grad_y ** 2 + 1.)
    uni_x = grad_x / A
    uni_y = grad_y / A
    uni_z = 1. / A
    vec_el = degree_lookat  # degree where light source looks at
    vec_az = degree  # degree of light source
    dx = np.cos(vec_el) * np.cos(vec_az)  # effect of light source on x-axis
    dy = np.cos(vec_el) * np.sin(vec_az)  # effect of light source on y-axis
    dz = np.sin(vec_el)  # effect of light source on z-axis
    b = 255 * (dx * uni_x + dy * uni_y + dz * uni_z)  # normalization effect of light source
    b = b.clip(0, 255)
    im = Image.fromarray(b.astype('uint8'))  # create new image
    im.save(result_image)

def main():
    indir = 'image'
    depth = 20
    degree_lookat = np.pi / 2.0
    degree = np.pi / 4
    exts = set('.jpg,.bmp,.png'.split(','))
    start_time = time.clock()
    for path in os.listdir(indir):
        shortname,ext = os.path.splitext(path)
        if ext not in exts:
            continue
        if re.findall("_HD",shortname):
            continue
        src_path = os.path.join(indir, path)
        dst_path = os.path.join(indir, shortname + "_HD"  + ext )
        image(src_path,dst_path,depths=depth,degree=degree, degree_lookat = degree_lookat)
    end_time = time.clock()
    print('runing time:' + str(end_time - start_time) + 's')

if __name__=="__main__":
    main()