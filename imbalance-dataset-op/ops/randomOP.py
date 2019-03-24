import os,shutil
import os.path as osp
import numpy as np
def _with_replace(origin_list, output_dir, req_num):
    labels = set([])
    for path_label in origin_list:
        path, label = path_label
        labels.add(label)
    for label in labels:
        os.makedirs(osp.join(output_dir,label))
    inds = np.randint(0,len(origin_list),size = req_num)
    for num, ind in enumerate(inds):
        src, label = origin_list[ind]
        dst = osp.join( osp.join(output_dir,label) , '{}'.format(num))
        shutil.copyfile(src,dst)

def sampling(input_file, output_dir, req_num):
    samples = []
    with open(input_file, 'r') as f:
        for line in f:
            path, label = line.strip().split(',')
            samples.append((path,label))
    _with_replace(samples, output_dir,req_num)
    return

def sampling_with_class(input_file, output_dir, req_num):
    samples = []
    labels = set([])
    with open(input_file, 'r') as f:
        for line in f:
            path, label = line.strip().split(',')
            samples.append((path,label))
            labels.add(label)
    for label in labels:
        samples_one_label = list( filter(lambda x: x[1] == label, samples) )
        req_num_one_label = len(samples_one_label) * req_num // len(samples)
        if req_num_one_label < 1:
            print("miss samples with label {} after sampling".format(label))
            continue
        _with_replace(samples_one_label,output_dir,req_num_one_label)
    return



