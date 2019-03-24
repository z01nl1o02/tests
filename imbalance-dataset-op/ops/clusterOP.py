import os,shutil
import os.path as osp
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from PIL import Image
from collections import defaultdict
from randomOP import _with_replace

def _run_cluster(origin_list, cluster_num = 8, batch_size=100,resize=(64,64)):
    clf = MiniBatchKMeans(n_clusters=cluster_num,batch_size=batch_size)
    def next_batch(allfiles,batch_size):
        imgs = []
        inds = []
        for ind,(path,label) in enumerate(allfiles):
            img = Image.open(path).convert("L")
            img = img.resize(size=resize,Image.ANTIALIAS)
            img = np.reshape(np.array(img),(1,-1)).astype(np.float32) / 255.0
            imgs.append(img)
            inds.append(ind)
            if len(imgs) >= batch_size:
                yield  np.vstack(imgs), inds
                imgs = []
                inds = []
        if len(inds) > 0:
            return np.vstack(imgs), inds
    for _,batch in next_batch(origin_list,batch_size):
        clf.partial_fit(batch)

    cluster_dict = defaultdict(list)
    for inds, batch in next_batch(origin_list, batch_size):
        Ys = clf.predict(batch)
        for y, ind in zip(Ys, inds):
            path,label = origin_list[ind]
            cluster_dict.setdefault(y,[]).append((path,label))
    return cluster_dict


def sampling(input_file, output_dir, req_num, resize = (96,96)):
    samples = []
    labels = set([])
    with open(input_file, 'r') as f:
        for line in f:
            path, label = line.strip().split(',')
            samples.append((path,label))
            labels.add(label)
    for label in labels:
        samples_one_label = list( filter(lambda x: x[1] == label, samples) )
        cluster_num = 8
        if len(samples_one_label) < 100:
            _with_replace(samples_one_label,output_dir,len(samples_one_label) * req_num // len(samples))
        else:
            cluster_info = _run_cluster(samples_one_label,cluster_num=cluster_num,batch_size=100,resize=resize)
            for cluster in cluster_info.keys():
                _with_replace(cluster_info[cluster],output_dir,len(cluster_info[cluster]) * req_num // len(samples))
    return
