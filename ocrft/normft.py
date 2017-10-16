import os,sys,pdb
import numpy as np
import math

normW = 32
normH = 64
normA = math.pi
def norm_one(infile,outfile):
    lines = []
    with open(infile,'rb') as f:
        for line in f:
            line = line.strip()
            if line == '':
                continue
            data = line.split('|')
            label = data[0]
            path = data[-1]
            line = [label]
            for d in data[1:-1]:
                cx,cy,ori = [np.float64(x) for x in d.split(',')]
                cx = cx / normW
                cy = cy / normH
                ori = ori / normA
                d = '%f,%f,%f'%(cx,cy,ori)
                line.append( d )
            line.append(path)
            lines.append('|'.join(line))
    with open(outfile,'wb') as f:
        f.writelines('\r\n'.join(lines))

def norm_all(indir,outdir):
    try:
        os.makedirs(outdir)
    except Exception,e:
        print e
        pass
    for txt in os.listdir(indir):
        src = os.path.join(indir,txt)
        dst = os.path.join(outdir,txt)
        norm_one(src,dst)

if __name__=="__main__":
    norm_all('feats','feat_norm')







