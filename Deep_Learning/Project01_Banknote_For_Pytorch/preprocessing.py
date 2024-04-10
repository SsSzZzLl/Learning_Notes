'''
Author: Szl
Date: 2024-04-10 10:32:42
LastEditors: Szl
LastEditTime: 2024-04-10 11:20:35
Description: 实现数据预处理脚本，同时完成数据集划分
正常的工程化发开场景中，该脚本往往只需要单独运行，所以是否采用函数或对象封装都可以
'''

# Package and Modules import statements
# -----------------------

import numpy as np
from config import HP
import os

# codings
# -----------------------

def data_split(data_path):
  
  '''
    :param data_path: 原始数据集的路径
    :return: 
  '''

  # 设置训练集、验证集、测试集划分比例为7：2：1
  trainset_ratio = 0.7
  devset_ratio = 0.2
  testset_ratio = 0.1
  
  # 设置随机种子
  np.random.seed(HP.seed)

  # 从给定的原始数据集路径加载原始数据集
  dataset = np.loadtxt(data_path, delimiter= ',')
  
  # 进行数据集的随机重新排列
  np.random.shuffle(dataset)
  
  # 获取样本数量
  n_items = dataset.shape[0]
  
  # 指定训练集样本数量
  trainset_num = int(trainset_ratio * n_items)
  
  # 指定验证集样本数量
  devset_num = int(devset_ratio * n_items)
  
  # 指定测试集样本数量
  testset_num = n_items - trainset_num - devset_num
  
  # 划分并生成训练集数据
  np.savetxt(os.path.join(HP.data_dir, 'train.txt'), dataset[: trainset_num], delimiter = ',')
  
  # 划分并生成验证集数据
  np.savetxt(os.path.join(HP.data_dir, 'dev.txt'), dataset[trainset_num : trainset_num + devset_num], delimiter = ',')
  
  # 划分并生成测试集数据
  np.savetxt(os.path.join(HP.data_dir, 'test.txt'), dataset[trainset_num + devset_num :], delimiter = ',')
  

# run test UseCase if current modules in main
# -----------------------

if __name__ == '__main__':
  data_split(HP.data_path)