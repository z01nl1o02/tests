1.输入图像尺寸不易太小,受制于pretrained net的限制
2.想保留细节,则content的层要靠近输入,否则要靠近输出
3.noise loss的权重提高到一定程度,有涂抹效果