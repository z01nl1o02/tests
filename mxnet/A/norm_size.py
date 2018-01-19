import os,sys,pdb
import cv2
root = 'origin'

try:
    os.makedirs('resized/cat')
    os.makedirs('resized/dog')
except Exception,e:
    pass

def do_resize(classname):
    indir = os.path.join(root,classname)
    outdir = os.path.join(root,classname)
    for jpg in os.listdir(indir):
        fname = os.path.join(indir,jpg)
        img = cv2.imread(fname,1)
        img = cv2.resize(img,(277,277))
        fname = os.path.join(outdir,jpg)
        cv2.imwrite(fname,img)

do_resize('cat')
do_resize('dog')
