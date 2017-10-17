import os,sys,pdb
import cv2

def run(mode):
    indir = 'images\\%s'%mode
    outdir = 'resized\\%s'%mode
    config = 'config\\config_resize.ini'
    for obj in os.listdir(indir):
        cmd = 'mkdir "%s"'%os.path.join(outdir,obj)
        os.system(cmd)
        cmd = 'tconvert.exe -indir="%s" -outdir="%s" -config="%s"'%(
                os.path.join(indir,obj),
                os.path.join(outdir,obj),
                config)
        os.system(cmd)


    indir = 'resized\\%s'%mode
    outdir = 'seg\\%s'%mode
    config = 'config\\config_otsu.ini'
    for obj in os.listdir(indir):
        cmd = 'mkdir "%s"'%os.path.join(outdir,obj)
        os.system(cmd)
        cmd = 'tconvert.exe -indir="%s" -outdir="%s" -config="%s"'%(
                os.path.join(indir,obj),
                os.path.join(outdir,obj),
                config)
        os.system(cmd)

    for folder in os.listdir('seg\\%s'%mode):
        indir = os.path.join('seg\\%s'%mode,folder)
        outdir =os.path.join('revert\\%s'%mode,folder)
        try:
            os.makedirs(outdir)
        except Exception,e:
            print e
        for name in os.listdir( indir ):
            img = cv2.imread(os.path.join(indir,name),0)
            img = 255 - img
            cv2.imwrite( os.path.join(outdir,name), img)


    indir = 'revert\\%s'%mode
    outdir = 'thin\\%s'%mode
    config = 'config\\config_thin.ini'
    for obj in os.listdir(indir):
        cmd = 'mkdir "%s"'%os.path.join(outdir,obj)
        os.system(cmd)
        cmd = 'tconvert.exe -indir="%s" -outdir="%s" -config="%s"'%(
                os.path.join(indir,obj),
                os.path.join(outdir,obj),
                config)
        os.system(cmd)

if __name__=="__main__":
    run('train')
    run('test')
