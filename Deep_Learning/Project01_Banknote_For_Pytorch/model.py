'''
Author: Szl
Date: 2024-04-10 10:34:27
LastEditors: Szl
LastEditTime: 2024-04-10 15:55:23
Description: 完成全连接神经网络的模型构建
'''

# Package and Modules import statements
# -----------------------

import torch
from torch import nn
from torch.nn import functional as F
from config import HP


# codings
# -----------------------

class BankNokeClassificationModel(nn.Module):
  
  '''
    构建全连接神经网络的模型构建
  '''

  def __init__(self):
    super().__init__()
    
    # 定义神经网络模型 - 依据layer_list超参指定的网络结构进行创建
    self.lineear_layer = nn.ModuleList([
      nn.Linear(in_features = in_dim, out_features = out_dim, bias = True)
      for in_dim, out_dim in zip(HP.layer_list[: -1], HP.layer_list[1 :])
    ]) # 一次性创建多层神经网络层
    
  # 完成前向计算
  def forward(self, input_x):
    
    '''
      重载前向计算方法
      :param input_x: 输入的一个batch_size的样本数据
      :return: y_hat
    '''

    for layer in self.lineear_layer:
      input_x = layer(input_x) # 此处等号左侧的input_x是第一神经网络层完成的线性变换的运算结果，需要给到一个激活函数进行非线性映射
      input_x = F.relu(input_x) # 此处采用的激活函数为relu激活
      
    return input_x # 最终的前向计算完成的y_hat预测结果

# run test UseCase if current modules in main
# -----------------------

if __name__ == '__main__':
  
  # 构建模型对象
  model = BankNokeClassificationModel() 
  
  # fake data
  x = torch.randn(size = (16, HP.in_features)).to(HP.device)
  y_pred = model(x)
  print(y_pred.size())