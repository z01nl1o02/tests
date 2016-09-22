import hashlib
import os,sys,pdb,cPickle
import argparse

def load_image_md5(indir):
    md2jpg = {}
    total = 0
    dup = 0
    for root,dnames,names in os.walk(indir):
        for name in names:
            fname = os.path.join(root, name)
            if os.path.isfile(fname) and 0 == cmp(os.path.splitext(fname)[-1],'.jpg'):
                fd = open(fname,'rb')
                md = hashlib.md5(fd.read()).hexdigest()
                total += 1
                if md not in md2jpg:
                    md2jpg[md] = [fname]
                else:
                    md2jpg[md].append(fname)
                    dup += 1
                if 0 == (total % 10000):
                    print '%d-%d,'%(dup,total),
                fd.close()
    print '%d-%d'%(dup,total)
    return md2jpg

def run(indir,outpkl):
    md2jpg = load_image_md5(indir)
    with open(outpkl, 'wb') as f:
        cPickle.dump(md2jpg, f)
    return    
if __name__=="__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('indir',help='input dir')
    ap.add_argument('outpkl',help='output pkl')
    args = ap.parse_args()
    run(args.indir, args.outpkl)

