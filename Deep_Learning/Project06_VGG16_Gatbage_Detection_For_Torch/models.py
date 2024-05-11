'''
Author: Szl
Date: 2024-05-11 17:26:00
LastEditors: Szl
LastEditTime: 2024-05-11 19:55:46
Description: 
'''
# Package and Modules import statements
# -----------------------

import torch
import torchsummary as summary


from torch import nn
from config import HP


# codings
# -----------------------

'''
  搭建模型
'''
class vgg16_diy(nn.Module):
  
  def __init__(self):
    
    super(vgg16_diy, self).__init__()
    
    # nn.Sequential是一个Sequential容器，模块将按照构造函数中传递的顺序添加到模块中
    
    # 卷积块1：224*224*64
    self.conv1 = nn.Sequential(
      nn.Conv2d(3, 64, kernel_size  = (3,3), stride  = (1, 1), padding = (1, 1)),
      nn.ReLU(),
      nn.Conv2d(64, 64, kernel_size = (3, 3), stride = (1, 1), padding = (1, 1)),
      nn.ReLU(), nn.MaxPool2d(kernel_size = (2, 2), stride = (2, 2)))
    
    # 卷积块2：112*112*128
    self.conv2 = nn.Sequential(
      nn.Conv2d(64,128,kernel_size = (3,3),stride = (1, 1),padding = (1, 1)),
      nn.ReLU(),
      nn.Conv2d(128,128,kernel_size = (3,3),stride = (1, 1),padding = (1, 1)),
      nn.ReLU(), nn.MaxPool2d(kernel_size = (2, 2), stride = (2, 2)))
    
    # 卷积块3：56*56*256
    self.conv3 = nn.Sequential(
      nn.Conv2d(128, 256, kernel_size = (3, 3), stride = (1, 1), padding = (1, 1)),
      nn.ReLU(),
      nn.Conv2d(256, 256, kernel_size = (3, 3), stride = (1, 1), padding = (1, 1)),
      nn.ReLU(),
      nn.Conv2d(256, 256, kernel_size = (3, 3), stride = (1, 1), padding = (1, 1)),
      nn.ReLU(),
      nn.MaxPool2d(kernel_size = (2, 2), stride = (2, 2)))
    
    # 卷积块4：28*28*512
    self.conv4 = nn.Sequential(
      nn.Conv2d(256, 512, kernel_size = (3, 3), stride = (1, 1), padding = (1, 1)),
      nn.ReLU(),
      nn.Conv2d(512, 512, kernel_size = (3, 3), stride = (1, 1), padding = (1, 1)),
      nn.ReLU(),
      nn.Conv2d(512, 512, kernel_size = (3, 3), stride = (1, 1), padding = (1, 1)),
      nn.ReLU(),
      nn.MaxPool2d(kernel_size = (2, 2), stride = (2, 2)))
    
    # 卷积块5：14*14*512
    self.conv5 = nn.Sequential(
      nn.Conv2d(512, 512, kernel_size = (3, 3), stride = (1, 1), padding = (1, 1)),
      nn.ReLU(),
      nn.Conv2d(512, 512, kernel_size = (3, 3), stride = (1, 1), padding = (1, 1)),
      nn.ReLU(),
      nn.Conv2d(512, 512, kernel_size = (3, 3), stride = (1, 1), padding = (1, 1)),
      nn.ReLU(),
      nn.MaxPool2d(kernel_size = (2, 2), stride = (2, 2)))
    
    # avgpool
    # self.avgpool = nn.AdaptiveAvgPool2d(output_size = (7, 7))
    
    # 全连接层，用于分类
    self.classifier = nn.Sequential(
      nn.Linear(in_features = 512*7*7, out_features = 4096),
      nn.ReLU(),
      # nn.Dropout(p = 0.5),
      nn.Linear(in_features = 4096, out_features = 4096, bias = True),
      nn.ReLU(),
      # nn.Dropout(p = 0.5),
      nn.Linear(in_features = 4096, out_features = 6)
    )

  # 前向传播
  def forward(self, x):
    
    x = self.conv1(x) # 卷积-BN-激活
    x = self.conv2(x)
    x = self.conv3(x)
    x = self.conv4(x)
    x = self.conv5(x)
    # x = self.avgpool(x)
    x = torch.flatten(x, start_dim = 1)
    x = self.classifier(x)

    return x
 
device = HP.device
# print("Using {} device".format(device))
 
model = vgg16_diy().to(device)

# print(model)

summary.summary(model, (3, 224, 224))
# run test UseCase if current modules in main
# -----------------------

if __name__ == '__main__':
  pass