import os,sys,cv2
import mxnet as mx
from mxnet import gluon
from mxnet import nd
from mxnet.gluon import utils
import numpy as np
import logging
import datetime
import random
from config import config
from dataset import JSDataset
from symbol.cifarnet import CIFARNET_QUICK
from collections import defaultdict,OrderedDict
from symbol.cifarnet import CIFARNET_QUICK,CIFARNET_BLOCK,CIFARNET_BLOCK_A
nowTime=datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

handleFile = logging.FileHandler(config.log,mode="wb")
handleFile.setFormatter(formatter)

handleConsole = logging.StreamHandler()
handleConsole.setFormatter(formatter)

logger = logging.getLogger()
logger.setLevel(level=logging.INFO)
logger.handles = []
logger.addHandler(handleFile)
logger.addHandler(handleConsole)


def get_classifer(max_sample_num):
    ctx = mx.gpu()
    crop_hw = (config.height, config.width)

    ds_train = JSDataset(config.train_root, fortrain=True, crop_hw = crop_hw,max_sample_num = max_sample_num)
    ds_test = JSDataset(config.test_root, fortrain=False, crop_hw=crop_hw, max_sample_num = max_sample_num)

    trainiter = gluon.data.DataLoader(ds_train,batch_size=config.batch_size, shuffle=True,last_batch="rollover")
    testiter = gluon.data.DataLoader(ds_test,batch_size=config.batch_size, shuffle=False,last_batch="rollover")

    logging.info("train num: {} test num: {}".format(len(ds_train), len(ds_test)))

    max_update = config.max_epoch * len(ds_train) // config.batch_size
    lr_sch = mx.lr_scheduler.PolyScheduler(max_update=max_update,base_lr=config.base_lr,pwr=1)

    net = CIFARNET_QUICK(class_num = config.class_num, ctx=ctx)

    if not (config.pretrained_model is None) and not (config.pretrained_model == ""):
        net.load_params(config.pretrained_model,ctx = ctx)
        logging.info("loading model {}".format(config.pretrained_model))

    trainer = mx.gluon.Trainer(net.collect_params(),optimizer=config.optimizer,optimizer_params={"learning_rate":config.base_lr})


    loss_ce = mx.gluon.loss.SoftmaxCrossEntropyLoss()

    class ACC_SHOW(object):
        def __init__(self,label_num):
            self.label_num = label_num
            self.axis = 1
            self.acc = {'total':0,'hit':0}
            self.acc_per_class = OrderedDict()
            for key in range(label_num):
                self.acc_per_class[key] = {'total':0,'hit':0}
            return
        def reset(self):
            self.acc = {'total':0,'hit':0}
            self.acc_per_class = OrderedDict()
            for key in range(self.label_num):
                self.acc_per_class[key] = {'total':0,'hit':0}
            return

        def update(self,preds, labels):
            if isinstance(preds[0],mx.nd.NDArray):
                preds = map(lambda pred: pred.asnumpy(),preds)
                labels = map(lambda label: label.asnumpy(),labels)
            for pred, label in zip(preds,labels):
                pred_label = np.argmax(pred,axis=self.axis)
                label = label.flatten().astype('int32')
                pred_label = pred_label.flatten().astype('int32')
                for p,l in zip(pred_label,label):
                    self.acc_per_class[l]['total'] += 1
                    self.acc['total'] += 1
                    if l == p:
                        self.acc_per_class[l]['hit'] += 1
                        self.acc['hit'] += 1
            return

        def _calc_acc(self,md):
            total = md['total']
            hit = md['hit']
            if total < 1:
                return 0
            return float(hit) / total

        def get_acc(self):
            return self._calc_acc(self.acc)

        def get(self):
            infos = ['acc {:.5} acc_per_class'.format( self._calc_acc(self.acc) )]
            for key in self.acc_per_class.keys():
                #print self.acc_per_class[key]
                infos.append('{:.3}'.format(self._calc_acc(self.acc_per_class[key])))
            return ' '.join(infos)

    class LOSS_SHOW(object):
        def __init__(self):
            self.loss_list = []

        def reset(self):
            self.loss_list = []

        def update(self, loss_list):
            if isinstance(loss_list[0],mx.nd.NDArray):
                loss_list = map(lambda loss: loss.asnumpy(), loss_list)
            loss = np.vstack(loss_list)
            #print loss.tolist()[0]
            self.loss_list.extend(loss.tolist()[0])

        def get(self):
            return "loss {:.5}".format( np.asarray(self.loss_list).mean()  )
    import pdb
    def show_gradient(net):
        return
        grads_list = []
        for block in net.layers:
            if not isinstance(block, CIFARNET_BLOCK) and not isinstance(block, CIFARNET_BLOCK_A):
                continue
            for layer in block.layers:
                if not isinstance(layer, gluon.nn.Conv2D):
                    continue
                grads = layer.weight.grad().as_in_context(mx.cpu()).asnumpy()
                grads_list.append(grads.mean())
                grads_list.append(grads.max())
                grads_list.append(grads.min())
        line = ['grads: ']
        for grads in grads_list:
            line.append( '%.6f'%grads )
        logging.info(','.join(line))
        return


    class TopAcc:
        def __init__(self):
            self.path = ""
            self.score = 0
        def update(self, path, score):
            if self.score < score:
                self.score = score
                self.path = path
            return
        def get_top(self):
            return self.path,self.score

    top_acc = TopAcc()

    loss_show = LOSS_SHOW()
    acc = ACC_SHOW( config.class_num )
    display_iter = len(ds_train) // (2 * config.batch_size )
    if display_iter < 1:
        display_iter = 1
    update = 0
    for epoch in range(config.max_epoch):
        acc.reset()
        loss_show.reset()
        for batch in trainiter:
            update += 1
            data, label = batch
            data_list, label_list = utils.split_and_load(data,ctx_list=[ctx]), utils.split_and_load(label,ctx_list=[ctx])
            with mx.autograd.record():
                pred_list = map(lambda data: net(data), data_list)
                loss_list = map(lambda (pred,label): loss_ce(pred,label), zip(pred_list,label_list))
            for loss in loss_list:
                loss.backward()
            trainer.step(config.batch_size)
            mx.nd.waitall()
            acc.update(labels = label_list,preds = pred_list)
            loss_show.update(loss_list)
            if 0 == (update % display_iter):
                logging.info("train update {} lr {} {} {}".format(update,trainer.learning_rate,loss_show.get(), acc.get()))
            trainer.set_learning_rate(lr_sch(update))
        acc.reset()
        show_gradient(net)
        loss_show.reset()
        for (data,label) in testiter:
            data_list,label_list = utils.split_and_load(data,[ctx]),utils.split_and_load(label, [ctx])
            pred_list = map(lambda data : net(data), data_list)
            loss_list = map(lambda (pred,label): loss_ce(pred,label), zip(pred_list,label_list))
            mx.nd.waitall()
            acc.update(labels = label_list, preds = pred_list)
            loss_show.update(loss_list)
        logging.info("test update {} epoch {} {} {}".format(update,epoch,loss_show.get(), acc.get()))
        if epoch % config.save_epoch_step == 0:
            net.save_params(config.model_path(epoch))
            top_acc.update( config.model_path(epoch), acc.get_acc() )
    net.save_params(config.model_path("last"))
    return top_acc.get_top()

