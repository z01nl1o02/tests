
[This looks like that: deep learning for interpretable image recognition](https://arxiv.org/abs/1806.10574)

# 代码
demo_cifar.py: 在cifar10上调优的几个成熟的网络结构
demo_cifar2.py: 在前者的基础上做了简化,主要方便测试新op,ProtoBlock


# 进度
* projection之前，train loss不断降低，projection之后，train loss一直升高
* R2的限制没法使用，一旦使用，train loss不断升高
* 下一步需要用全部数据做测试，上述问题可能和weight训练不足，直接projection引发了问题