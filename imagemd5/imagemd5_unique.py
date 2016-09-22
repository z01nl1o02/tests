import os,sys,pdb,cPickle
import hashlib
import argparse

def run(md2jpg_pkl,outbatch,outdir):
    with open(md2jpg_pkl,'rb') as f:
        md2jpg = cPickle.load(f)
    line_list = ['mkdir "%s"'%outdir]
    for md in md2jpg.keys():
        line_list.append(  'move "%s" "%s"'%(md2jpg[md][0],outdir) )
    line = '\r\n'.join(line_list)
    with open(outbatch,'wb') as f:
        f.writelines(line)

if  __name__=="__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('md2jpg',help='pkl storing md2jpg')
    ap.add_argument('outbatch',help='output batch')
    args = ap.parse_args()
    outdir = os.path.splitext(args.outbatch)[0]
    run(args.md2jpg,args.outbatch,outdir)

