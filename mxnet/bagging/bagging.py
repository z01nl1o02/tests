import train
import shutil,os
from symbol.cifarnet import CIFARNET_QUICK
from config import config
import mxnet as mx
from mxnet import gluon
from dataset import JSDataset
import numpy as np

import copy
sample_num = 10000

flag_train = False

if flag_train:
    for bagging_num in range(10):
        path,score = train.get_classifer(sample_num)
        os.rename(path, 'bagging_{}.params'.format(bagging_num))

from collections import defaultdict
class ClassificationResult:
    def __init__(self):
        self.clf2score = defaultdict(list)
        return
    def update(self,clf_idx, pred_score, target_label):
       # if isinstance(pred_score,list):
        pred_score = pred_score[0].as_in_context(mx.cpu()).asnumpy()
        target_label = target_label[0].as_in_context(mx.cpu()).asnumpy()[0]
        self.clf2score[clf_idx].append((pred_score[0], pred_score[1],int(target_label)))
        return
    def _get_acc(self,scores):
        total = len(scores)
        correct = np.asarray(map(lambda x:x[int(x[-1])] > x[1-int(x[-1])], scores)).sum()
        return  float(correct)/total
    def _ensemble_clfs(self):
        row = len(self.clf2score[0])
        col = 3
        res = np.zeros( (row,col) )
        for clf_idx in self.clf2score.keys():
            res += np.asarray(self.clf2score[clf_idx])
        res /= len(self.clf2score)
        for y in range(row):
            self.clf2score[-1].append( res[y].tolist() )
        return
    def get(self,clf_idx=-1):
        if clf_idx >= 0:
            return 'clf {} acc {}'.format(clf_idx, self._get_acc(self.clf2score[clf_idx]))
        else:
            self._ensemble_clfs()
            clf_idx = -1
            return 'clf {} acc {}'.format(clf_idx, self._get_acc(self.clf2score[clf_idx]))

from tqdm import tqdm
if not flag_train:
    num_clf = 10
    num_sample = -1
    clf_results = ClassificationResult()
    ctx = mx.gpu()
    clf_dict = {}
    for k in range(num_clf):
        clf_dict[k] = CIFARNET_QUICK(class_num=config.class_num, ctx=ctx)
        net_path = 'bagging_%d.params'%k
        clf_dict[k].load_parameters(net_path, ctx=ctx)
    ds_test = JSDataset(config.test_root, fortrain=False, crop_hw=(config.height, config.width), max_sample_num = num_sample)
    data_iter = gluon.data.DataLoader(ds_test,batch_size=1, shuffle=False,last_batch="rollover")
    for batch in tqdm(data_iter):
        data,label = batch
        data = data.as_in_context(ctx)
        for clf_idx in clf_dict.keys():
            pred_score = clf_dict[clf_idx](data)
            clf_results.update(clf_idx,pred_score,label)
    for clf_idx in clf_dict.keys():
        print clf_results.get(clf_idx)
    print clf_results.get(-1)


