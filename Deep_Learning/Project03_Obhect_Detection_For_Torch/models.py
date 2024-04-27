# Package and Modules import statements
# -----------------------

import torch
from torch import nn
from torch.nn import functional as F
from config import HP
from loss import Mish, mish

# codings
# -----------------------

class DSConv2d(nn.Module):
  
  def __init__(self, in_channels, out_channels, kernel_size):
    
    super().__init__()
    
    # 打一个断言，确保kernel_size % 2 == 1 - 确保kernel_size是奇数，否则不能进行后续的卷积操作
    assert kernel_size % 2 == 1, "needed!"
    
    # 先创建分组卷积
    self.depth_conv = nn.Conv2d(
      in_channels = in_channels,
      out_channels = in_channels,
      kernel_size = (kernel_size, kernel_size),
      padding = (kernel_size // 2, kernel_size // 2),
      groups = in_channels
    )

    # 再创建逐点卷积
    self.point_conv = nn.Conv2d(
      in_channels = in_channels,
      out_channels = out_channels,
      kernel_size = (1, 1)
    )
  
  # 前向计算
  def forward(self, input_x):
    return self.point_conv(self.depth_conv(input_x))
    
# 模型构建第二步：实现残差块
class TrailBlock(nn.Module):
  
  '''
    构建残差块
  '''

  def __init__(self, in_channels):
    super().__init__()
    
    # 左分支
    self.left_flow = nn.Sequential(
      
      # 1.点卷积
      nn.Conv2d(in_channels = in_channels, out_channels = in_channels, kernel_size = (1, 1)),
      nn.BatchNorm2d(in_channels),
      Mish(),
      
      # 2.深度可分离卷积
      DSConv2d(in_channels = in_channels, out_channels = in_channels, kernel_size = 3),
      nn.BatchNorm2d(in_channels),
      Mish(),
      
      # 3.7 * 7卷积核
      nn.Conv2d(in_channels = in_channels, out_channels = in_channels, kernel_size = (7, 7), padding = (7 // 2, 7 // 2))
    )
    
    # 右分支
    self.right_flow = nn.Sequential(
      # 1.7 * 7卷积核
      nn.Conv2d(in_channels = in_channels, out_channels = in_channels, kernel_size = (7, 7), padding = (7 // 2, 7 // 2)),
      nn.BatchNorm2d(in_channels),
      Mish(),
      
      # 2.深度可分离卷积
      DSConv2d(in_channels = in_channels, out_channels = in_channels, kernel_size = 3),
      nn.BatchNorm2d(in_channels),
      Mish(),
      
      # 3.点卷积
      nn.Conv2d(in_channels = in_channels, out_channels = in_channels, kernel_size = (1, 1)),
      nn.BatchNorm2d(in_channels),
      Mish()
    )
  
  # 前向计算
  def forward(self, input_x):
    return mish(self.left_flow(input_x) + self.right_flow(input_x) + input_x)
 
# 模型构建第三步：完成最终的深度残差神经网络模型的构建
class Net(nn.Module):
  
  '''
    构建深度残差神经网络模型
  '''

  def __init__(self):
    super().__init__()

    # 1.构建卷积层，负责高维图像特征提取
    
    self.mtn_conv = nn.Sequential(
      
      # 3 * 3卷积核，输出通道数64
      nn.Conv2d(in_channels = HP.data_channels, out_channels = 64, kernel_size = (3, 3), padding = (3 // 2, 3 // 2)),
      nn.BatchNorm2d(64),
      Mish(),
      
      # 第一个输入通道数为64，输出通道数为64的残差块
      TrailBlock(in_channels = 64),
      nn.MaxPool2d(kernel_size = (2, 2), stride = (2, 2)),
      
      # 3 * 3 卷积核，输出通道数128
      nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = (3, 3), padding = (3 // 2, 3 // 2)),
      nn.BatchNorm2d(128),
      Mish(),
      
      # 第二个输入通道数为128，输出通道数为128的残差块
      TrailBlock(in_channels = 128),
      nn.MaxPool2d(kernel_size = (2, 2), stride = (2, 2)),
      
      # 3 * 3 卷积核，输出通道数256
      nn.Conv2d(in_channels = 128, out_channels = 256, kernel_size = (3, 3), padding = (3 // 2, 3 // 2)),
      nn.BatchNorm2d(256),
      Mish(),
      
      # 第三个输入通道数为256，输出通道数为256的残差块
      TrailBlock(in_channels = 256),
      nn.MaxPool2d(kernel_size = (2, 2), stride = (2, 2)),
      
      # 再次接三个输入输出为256的残差块
      TrailBlock(in_channels = 256),
      TrailBlock(in_channels = 256),
      TrailBlock(in_channels = 256),
      nn.MaxPool2d(kernel_size = (2, 2), stride = (2, 2)), # [256, 7, 7]
    )
    
    # 2. 构建全连接层，完成图像分类
    self.mtn_fc = nn.Sequential(
      # 第一全连接，输入为256 * 7 * 7，输出为2048
      nn.Linear(in_features = 256 * 7 * 7, out_features = 2048),
      Mish(),
      nn.Dropout(HP.fc_drop_prob),
      
      # 第二全连接，输入为2048，输出为1024
      nn.Linear(in_features = 2048, out_features = 1024),
      Mish(),
      nn.Dropout(HP.fc_drop_prob),
      
      # 最终输出层，输入为1024，输出为6个类别
      nn.Linear(in_features = 1024, out_features = HP.classes_num)
    )
   
  # 前向计算  
  def forward(self, input_x):
    out = self.mtn_conv(input_x)
    return self.mtn_fc(out.view(input_x.size(0), -1))    
    
# run test UseCase if current modules in main
# -----------------------

if __name__ == '__main__':
  pass