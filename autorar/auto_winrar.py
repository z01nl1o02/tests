import os,sys
from subprocess import Popen
import argparse
import numpy as np

def get_sub_folder_list(root):
    folder_list = []
    objs = os.listdir(root)
    for obj in objs:
        fname = os.path.join(root,obj)
        folder_list.append(fname)
    return folder_list

def split_folder_list(folder_list,N):
    lists = [[] for k in range(N)]
    k = 0
    for folder in folder_list:
        lists[k].append(folder)
        k += 1
        if k >= N:
            k = 0
    return lists
    
def save_list(filename, folder_list):
    line = "\r\n".join(folder_list)
    with open(filename, 'wb') as f:
        f.writelines(line)
    return

def call_winrar(filelist, target_base_name):
    cmd = 'winrar a -ag -r %s @%s'%(target_base_name, filelist)
    p = Popen(cmd)
    return p.wait()

def run(root, N, outdir):
    lists = split_folder_list( get_sub_folder_list(root), N)
    listfile = 'list.txt'
    for k, folder_list in enumerate(lists):
        save_list(listfile, folder_list)
        ret = call_winrar(listfile, os.path.join(outdir,'%.6d_.rar'%k))
        print '%d end with %d'%(k+1, ret)
    return

if __name__=="__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('root', help = 'folder with subdirs')
    ap.add_argument('-N', default=10, help='splited number',type=np.int64)
    ap.add_argument('outdir',help='output folder')
    args = ap.parse_args()
    run(args.root, args.N, args.outdir)


