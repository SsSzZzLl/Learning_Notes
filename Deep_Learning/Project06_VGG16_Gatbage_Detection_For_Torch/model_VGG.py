'''
Author: Szl
Date: 2024-05-11 19:59:23
LastEditors: Szl
LastEditTime: 2024-05-11 20:02:19
Description: 
'''
# Package and Modules import statements
# -----------------------

import torch.nn as nn
import torchsummary as summary

from config import HP
from torchvision.models import vgg16


# codings
# -----------------------

'''
  搭建模型
'''
device = HP.device
# print("Using {} device".format(device))
 
# 加载预训练模型，并且对模型进行微调
model = vgg16(pretrained = True).to(device) # 加载预训练的vgg16
 
for param in model.parameters():
  param.requires_grad = False # 冻结模型的参数，这样在训练的时候只训练最后一层的参数
 
# 修改classfier模块的第六层（即：(6):Linear(in_features=4096, out_features=2, bias=True))
# 注意下面打印出来的模型
model.classifier._modules['6'] = nn.Linear(4096, 6) # 修改vgg-16模型最后一层全连接层，输出目标类别个数
model.to(device)
# print(model)
 
'''
  查看模型详情
'''
# 统计模型参数量以及其他指标

# summary.summary(model, (3, 224, 224))


# run test UseCase if current modules in main
# -----------------------

if __name__ == '__main__':
  pass