'''
Author: Szl
Date: 2024-04-10 10:34:40
LastEditors: Szl
LastEditTime: 2024-04-12 19:02:26
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

from loss import get_loss_func
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
  criterion = get_loss_func()
  
  # 4.定义优化器
  opt = optim.Adam(model.parameters(), lr = HP.lr)
  
  # 5.加载数据集 - 先加载训练集数据，然后封装为dataloader对象
  trainset = BankNoteDataset(HP.trainset_path)
  train_loader = DataLoader(trainset, batch_size = HP.batch_size, shuffle = True, drop_last = True)
  
  # 加载验证集数据
  devset = BankNoteDataset(HP.devset_path)
  dev_loader = DataLoader(devset, batch_size = HP.batch_size, shuffle = True, drop_last = True)
  
  # 指定初始的训练轮数和步数
  start_epoch, step = 0, 0
  
  # 6.思考：是否可以确保模型训练一次开始，一定可以正常完成所有训练？能否确保模型训练中途不存在中断情况，既然无法确保模型一定从头训练到结束，那么如何实现训练的断点续传？
  
  # 7.先创建命令行参数传递对象
  parser = ArgumentParser(description = 'Model Training')
  
  # 8.添加命令行参数传递的细节
  parser.add_argument(
     '--c', # 表示该需要在命令行中传递的参数需要依托命令行中一个什么样的选项来触发
     default = None,
     type = str,
     help = 'train from scratch or resume training'
  )
  args = parser.parse_args()
  
  # 9.参数准备好后，训练是否从断点处开始，就依据命令行中是否携带了参数为标准
  if args.c: # 如果本次开启训练任务的python命令行脚本中携带了指定的参数
    
    # 表示此时需要从断点处重新恢复上一次的训练
    checkpoint = torch.load(args.c)

    # 恢复参数的更新
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # 继续将恢复出来的参数更新到优化器
    opt.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # 恢复上一次中断训练时所完成训练的轮数，接下来从这个轮数开始继续向后训练
    start_epoch = checkpoint['epoch']
    
    print('Resume from epoch {}'.format(args.c))
  
  else:
    print('Training from scratch!')
    
  # 开始进行模型训练
  # 注意：torch为我们封装了模型进入训练状态，进入验证状态的方法，开启训练前需要确保模型加入训练状态
  model.train()
  
  
    
  
  

# run test UseCase if current modules in main
# -----------------------