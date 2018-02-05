import os,sys,pdb
from PIL import Image
import argparse
import numpy as np

def main(num,numX,numY):
    result = None
    x = 0
    y = 0
    for k in range(num):
        img = Image.open('%d.png'%k)
        width,height = img.size
        if result is None:
            result = Image.new('RGBA',(width*numX,height*numY))
        result.paste(img,(x * width, y * height, (x+1)*width, (y+1)*height))
        x += 1
        if x >= numX:
            x = 0
            y += 1
    result.save("merged.png",quality=100)

if __name__=="__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('num',help="total png number",type=np.int64)
    ap.add_argument('numX',help="total png number along x",type=np.int64)
    ap.add_argument('numY',help="total png number along y",type=np.int64)
    args = ap.parse_args()
    main(args.num, args.numX, args.numY)
