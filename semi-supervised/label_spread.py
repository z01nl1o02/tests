

#https://scikit-learn.org/stable/modules/label_propagation.html#semi-supervised
#Learning with Local and Global Consistency


import numpy as np
from PIL import Image
import random

from scipy import stats

from sklearn.semi_supervised import label_propagation

import os

input_dir = 'digits'
datafile = "dataset.txt"

def gen_dataset(root,outfile):
    lines = []
    for rdir,pdir,names in os.walk(root):
        for name in names:
            path = os.path.join(rdir,name)
            label = int(path.split(os.path.sep)[-2])
            if random.randint(0,10) > 2:
                label = -1
            img = Image.open(path).convert("L")
            data = np.reshape(np.asarray(img),64).tolist()
            line = ["%d" % k for k in data]
            line.append("%d" % label)
            line.append(path)
            lines.append(' '.join(line))
    with open(outfile,'w') as f:
        f.write('\n'.join(lines))




def load_dataset(filepath):
    X, Y, paths= [], [], []
    with open(filepath,'r') as f:
        for line in f:
            data = line.strip().split(' ')
            path = data[-1]
            y = float(data[-2]) #-1 for unk label
            x = [float(d) for d in data[0:-2]]
            Y.append(y)
            X.append(x)
            paths.append(path)
    X = np.vstack(X)
    unique_Y = list(set(Y))
    print('total label: ',len(unique_Y))
    for y in unique_Y:
        num = len( list(filter(lambda d: d == y, Y)) )
        print("(",y, ",",num,"),",end="\t")
    print('')
    return X, Y, paths

gen_dataset(input_dir, datafile)
X,Y,paths = load_dataset(datafile)


unlabeled_set = [k for k in range(len(Y)) if Y[k] == -1]

# #############################################################################
# Learn with LabelSpreading
lp_model = label_propagation.LabelSpreading(gamma=0.25, max_iter=5)
lp_model.fit(X, Y)

#predicted_labels = lp_model.transduction_[unlabeled_set]

lines = []
labels = set([])
for idx in unlabeled_set:
    y,path = lp_model.transduction_[idx], paths[idx]
    outdir = os.path.join('output','%d'%int(y))
    if y not in labels:
        lines.append('mkdir %s'%outdir)
        labels.add(y)
    lines.append("copy {} {}".format(path,outdir))
with open("classify.bat","w") as f:
    f.write('\n'.join(lines))

# #############################################################################
# Calculate uncertainty values for each transduced distribution
pred_entropies = stats.distributions.entropy(lp_model.label_distributions_.T)

# #############################################################################
# Pick the top 10 most uncertain labels
uncertainty_index = np.argsort(pred_entropies)[-10:]

lines = []
labels = set([])
for idx in uncertainty_index:
    y,path = lp_model.transduction_[idx], paths[idx]
    outdir = os.path.join('uncertainty','%d'%int(y))
    if y not in labels:
        lines.append('mkdir %s'%outdir)
        labels.add(y)
    lines.append("copy {} {}".format(path,outdir))
with open("uncertainty.bat","w") as f:
    f.write('\n'.join(lines))

