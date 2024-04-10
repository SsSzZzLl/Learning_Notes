'''
Author: Szl
Date: 2024-04-10 10:33:29
LastEditors: Szl
LastEditTime: 2024-04-10 16:26:14
Description: 使用dataloader对象实现数据集按batch_size进行批量加载
'''

# Package and Modules import statements
# -----------------------

import torch
from torch.utils.data import DataLoader
from config import HP
import numpy as np

# codings
# -----------------------

class BankNoteDateset(torch.utils.data.Dataset):
  
  '''
    实现dataloader对象完成数据集按batch_size进行批量加载
  '''
  
  def __init__(self, data_path):

    # 根据传入的数据集路径加载数据 - 返回的是一个ndarray数组
    self.dataset = np.loadtxt(data_path, delimiter = ',')
    
  # 重写getitem方法，用于获取加载的一个batch_size大小的数据集
  def __getitem__(self, idx):
    
    '''
      根据指定的batch_size获取一个batch_size大小的数据
      :param idx: batch_size大小
      :return: tensor(x), tensor(y)
    '''
    
    # 获取一个batch_size数据
    item = self.dataset[idx]
    
    # 从一个batch_szie的数据中切出特征和标签 - 此时的x，y也是一个ndarray数组
    x, y = item[: HP.in_features], item[HP.in_features:]
    
    # 接下来使用torch来完成模型的构建和训练，需要输入的数据类型是tensor张量，并非ndarray，x和y切分出来后还要将其转换为tensor类型
    return torch.Tensor(x).float().to(HP.device), torch.Tensor(y).squeeze().long().to(HP.device)
    
  # 重写len方法，用于获取一个batch_size下的样本个数
  def __len__(self):
    return self.dataset.shape[0]

# run test UseCase if current modules in main
# -----------------------

if __name__ == '__main__':
  
  # 实例化当前类创建对象
  banknote_dataset = BankNoteDateset(HP.testset_path) # 使用测试集测试
  
  # 使用dataloader对象封装数据集加载对象
  banknote_dataloader = DataLoader(banknote_dataset, batch_size = 16, shuffle = True, drop_last = True)
  
  for batch in banknote_dataloader:
    x, y = batch
    print(x)
    print(y)
    break