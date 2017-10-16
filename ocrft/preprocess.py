import os,sys,pdb
import cv2

indir = 'images'
outdir = 'resized'
config = 'config\\config_resize.ini'
for obj in os.listdir(indir):
    cmd = 'mkdir "%s"'%os.path.join(outdir,obj)
    os.system(cmd)
    cmd = 'tconvert.exe -indir="%s" -outdir="%s" -config="%s"'%(
            os.path.join(indir,obj),
            os.path.join(outdir,obj),
            config)
    os.system(cmd)




indir = 'resized'
outdir = 'seg'
config = 'config\\config_otsu.ini'
for obj in os.listdir(indir):
    cmd = 'mkdir "%s"'%os.path.join(outdir,obj)
    os.system(cmd)
    cmd = 'tconvert.exe -indir="%s" -outdir="%s" -config="%s"'%(
            os.path.join(indir,obj),
            os.path.join(outdir,obj),
            config)
    os.system(cmd)

for folder in os.listdir('seg'):
    indir = os.path.join('seg',folder)
    outdir =os.path.join('revert',folder)
    try:
        os.makedirs(outdir)
    except Exception,e:
        print e
    for name in os.listdir( indir ):
        img = cv2.imread(os.path.join(indir,name),0)
        img = 255 - img
        cv2.imwrite( os.path.join(outdir,name), img)


indir = 'revert'
outdir = 'thin'
config = 'config\\config_thin.ini'
for obj in os.listdir(indir):
    cmd = 'mkdir "%s"'%os.path.join(outdir,obj)
    os.system(cmd)
    cmd = 'tconvert.exe -indir="%s" -outdir="%s" -config="%s"'%(
            os.path.join(indir,obj),
            os.path.join(outdir,obj),
            config)
    os.system(cmd)


