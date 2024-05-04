'''
Author: Szl
Date: 2024-04-29 13:43:10
LastEditors: Szl
LastEditTime: 2024-04-30 21:53:47
Description: 
'''
# Package and Modules import statements
# -----------------------

import torch
import torchvision.models as models

from torch import nn
from config import HP


# codings
# -----------------------

# 使用预训练好的ResNet50模型
# net = models.resnet50()
# net.load_state_dict(torch.load("./model/resnet50-11ad3fa6.pth"))
net = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

class GarbageDataSet(nn.Module):
  
  def __init__(self, net):
    super(GarbageDataSet, self).__init__()
    
    # resnet50
    self.net = net
    self.relu = nn.ReLU()
    self.dropout = nn.Dropout(0.1)
    self.fc = nn.Linear(1024, 6) # 全连接层1024个输入，6个输出
    self.output = nn.Softmax(dim=1) # 输出层使用Softmax激活函数

  def forward(self, x):
    x = self.net(x)
    x = self.relu(x)
    x = self.dropout(x)
    x = self.fc(x)
    x = self.output(x)
    return x
  
model = GarbageDataSet(net)


# run test UseCase if current modules in main
# -----------------------

if __name__ == '__main__':
  pass