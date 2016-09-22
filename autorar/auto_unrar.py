import os,sys
from subprocess import Popen
import argparse
import numpy as np


def unzip_rar_in_folder(root,target_ext,outdir):
    rar_list = []
    rars = os.listdir(root)
    for rar in rars:
        sname,ext = os.path.splitext(rar)
        if 0 != cmp(ext,'.zip') and 0 != cmp(ext,'.rar'):
            continue
        fname = os.path.join(root,rar)
        #keep folder structure in zip/rar
        #cmd = 'winrar x "%s" %s "%s\\%s\\"'%(fname, target_ext, root,sname)
        #ignore folder structure in zip/rar
        #cmd = 'winrar e "%s" %s "%s\\%s\\"'%(fname, target_ext, root,sname)
        cmd = 'winrar x "%s" %s "%s\\"'%(fname, target_ext, outdir)
        print cmd
        p = Popen(cmd)
        p.wait()

def run(root,target_ext,outdir):
    if 0 == cmp(root,'.'):
        print 'not support . as root'
        return
    unzip_rar_in_folder(root,target_ext,outdir)
    return

if __name__=="__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('root', help = 'folder with rars')
    ap.add_argument('-ext',default='*.*',help='target ext in zip/rar')
    ap.add_argument('outdir',help='output dir')
    args = ap.parse_args()
    run(args.root,args.ext,args.outdir)


