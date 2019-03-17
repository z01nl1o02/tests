import os,sys,shutil,random
indir = '0'
label_path = 'cluster_predict.txt'
outdir = 'cluster'
ratio_for_train = 0.66
labels = set([])

with open(label_path,'r') as f:
    for line in f:
        name,label = line.strip().split(',')
        src_path = os.path.join(indir,name)
        if label not in labels:
            labels.add(label)
            os.makedirs(os.path.join(outdir,'train',label))
            os.makedirs(os.path.join(outdir,'test',label))
        if random.random() < ratio_for_train: 
            shutil.copy(src_path, os.path.join(outdir,'train',label))
        else:
            shutil.copy(src_path, os.path.join(outdir,'test',label))

