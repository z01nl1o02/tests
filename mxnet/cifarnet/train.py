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
from dataset import DS_CIFAR
from symbol.cifarnet import CIFARNET

nowTime=datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
if not os.path.exists('log'):
    os.makedirs('log')
handleFile = logging.FileHandler("log/%s.txt"%nowTime,mode="wb")
handleFile.setFormatter(formatter)

handleConsole = logging.StreamHandler()
handleConsole.setFormatter(formatter)

logger = logging.getLogger()
logger.setLevel(level=logging.INFO)
logger.handles = []
logger.addHandler(handleFile)
logger.addHandler(handleConsole)



ctx = mx.gpu()

crop_hw = (config.height, config.width)
ds_train = DS_CIFAR(config.train_root, fortrain=True, crop_hw = crop_hw)
ds_test = DS_CIFAR(config.test_root, fortrain=False, crop_hw=crop_hw)

trainiter = gluon.data.DataLoader(ds_train,batch_size=config.batch_size, shuffle=True,last_batch="discard")
testiter = gluon.data.DataLoader(ds_test,batch_size=config.batch_size, shuffle=False,last_batch="discard")

logging.info("train num: {} test num: {}".format(len(ds_train), len(ds_test)))

max_update = config.max_epoch * len(ds_train) // config.batch_size
lr_sch = mx.lr_scheduler.PolyScheduler(max_update=max_update,base_lr=config.base_lr,pwr=1)



net = CIFARNET(class_num = config.class_num, ctx=ctx)
trainer = mx.gluon.Trainer(net.collect_params(),optimizer="adam",optimizer_params={"learning_rate":config.base_lr})


loss_ce = mx.gluon.loss.SoftmaxCrossEntropyLoss()
acc = mx.metric.Accuracy()
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
    def __str__(self):
        return "('loss', {})".format( np.asarray(self.loss_list).mean()  )

loss_show = LOSS_SHOW()

display_iter = len(ds_train) // (5 * config.batch_size )
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
            logging.info("train update {} lr {} {} {}".format(update,trainer.learning_rate,str(loss_show), acc.get()))
    acc.reset()
    loss_show.reset()
    for (data,label) in testiter:
        data_list,label_list = utils.split_and_load(data,[ctx]),utils.split_and_load(label, [ctx])
        pred_list = map(lambda data : net(data), data_list)
        loss_list = map(lambda (pred,label): loss_ce(pred,label), zip(pred_list,label_list))
        mx.nd.waitall()
        acc.update(labels = label_list, preds = pred_list)
        loss_show.update(loss_list)
    logging.info("test update {} epoch {} {} {}".format(update,epoch,str(loss_show), acc.get()))
    trainer.set_learning_rate(lr_sch(update))


