import os,sys,shutil,cv2,pickle
import random
from sklearn.cluster import MiniBatchKMeans as MBKMeans
import numpy as np

indir = '0'
cluster_number = 10
batch_size = cluster_number * 10
dim = 71*71
def get_batch():
    batch_data = []
    for name in os.listdir(indir):
        img = cv2.imread(os.path.join(indir,name),0)
        img = np.float32(img)/255.0
        batch_data.append(img.flatten())
        if len(batch_data) >= 100:
            yield np.vstack(batch_data)
            batch_data = []


kms = MBKMeans(n_clusters=cluster_number, batch_size = batch_size)
for batch in get_batch():
    print(batch.shape)
    kms.partial_fit(batch)

with open("cluster.pkl",'wb') as f:
    pickle.dump(kms,f)

lines = []
for name in os.listdir(indir):
    img = cv2.imread(os.path.join(indir,name),0)
    img = (np.float32(img)/255.0).flatten()
    img = np.vstack([img])
    ypred = kms.predict(img)[0]
    lines.append( ','.join([name,str(ypred)]))

with open('cluster_predict.txt','w') as f:
    f.write('\n'.join(lines))






