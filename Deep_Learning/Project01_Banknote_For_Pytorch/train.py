'''
Author: Szl
Date: 2024-04-10 10:34:40
LastEditors: Szl
LastEditTime: 2024-04-10 17:05:29
Description: 实现模型的训练过程，要求记录训练日志且训练过程中完成模型的依次训练保存
'''

# Package and Modules import statements
# -----------------------

import os
import random
import numpy as np
import torch
import torch.optim as optim # 导入需要的优化器对象
import torch.nn as nn

from model import BankNokeClassificationModel
from config import HP
from dataset_banknote_dataloader import BankNoteDataset
from torch.utils.data import DataLoader
from argparse import ArgumentParser # 整个对象允许在通过命令行的方式启动python脚本时，运行额外在命令行中传递一些参数 - python3 xxx.py
from tensorboardX import SummaryWriter # tensorboardX是一个专用于机器学习模型训练成果的可视化工具，会通过读取记录下来的训练日志来可视化完整的loss训练下降过程
# SummaryWriter可以实时记录训练日志并以行缓冲的方式实时向日志文件中写入日志数据

# codings
# -----------------------

# 创建日志记录对象
logger = SummaryWriter('./log')

# 设置所需的基本随机种子对象
torch.manual_seed(HP.seed)
torch.cuda.manual_seed(HP.seed)
random.seed(HP.seed)
np.random.seed(HP.seed)

def train():
  
  '''
    实现模型的训练过程
    :return: 
  '''

  # 1.构建模型
  model = BankNokeClassificationModel()
  
  # 2.模型送到指定硬件算力进行计算
  model = model.to(HP.device)
  
  # 3.定义损失函数
  loss_fn = nn.CrossEntropyLoss()
  
  
  

# run test UseCase if current modules in main
# -----------------------