'''
Author: Szl
Date: 2024-05-07 08:07:22
LastEditors: Szl
LastEditTime: 2024-05-07 08:21:26
Description: 
'''
import os
import torch
from config import HP
import random
import numpy as np
from torch import nn
from torch import optim
from dataloader import trainLoader, valLoader
from utils import evaluate, save_checkpoint
from models import model as GarbageDataSet
from argparse import ArgumentParser # 整个对象允许在通过命令行的方式启动python脚本时，运行额外在命令行中传递一些参数 - python3 xxx.py
from tensorboardX import SummaryWriter # tensorboardX是一个专用于机器学习模型训练成果的可视化工具，会通过读取记录下来的训练日志来可视化完整的loss训练下降过程

# 创建日志记录对象
logger = SummaryWriter('./log')

# 设置所需的基本随机种子对象
torch.manual_seed(HP.seed)
torch.cuda.manual_seed(HP.seed)
random.seed(HP.seed)
np.random.seed(HP.seed)

# 1. 获取模型
model = GarbageDataSet
model.to(HP.device) # 模型传入cuda

# 2. 定义损失函数
criterion = nn.CrossEntropyLoss() # 交叉熵损失函数

# 3. 定义优化器
optimizer = optim.Adam(model.parameters(), lr = 0.001) # 自适应矩估计优化



def train():
  
  '''
    实现模型的训练过程
    :return: 
  '''
  
  # 指定初始的训练轮数和步数
  start_epoch, step = 0, 0
  
  # 4.思考：是否可以确保模型训练一次开始，一定可以正常完成所有训练？能否确保模型训练中途不存在中断情况，既然无法确保模型一定从头训练到结束，那么如何实现训练的断点续传？
  
  # 5.先创建命令行参数传递对象
  parser = ArgumentParser(description = 'Model Training')
  
  # 6.添加命令行参数传递的细节
  parser.add_argument(
     '--c', # 表示该需要在命令行中传递的参数需要依托命令行中一个什么样的选项来触发
     default = None,
     type = str,
     help = 'train from scratch or resume training'
  )
  args = parser.parse_args()
  
  # 7.参数准备好后，训练是否从断点处开始，就依据命令行中是否携带了参数为标准
  if args.c: # 如果本次开启训练任务的python命令行脚本中携带了指定的参数
    
    # 表示此时需要从断点处重新恢复上一次的训练
    checkpoint = torch.load(args.c)

    # 恢复参数的更新
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # 继续将恢复出来的参数更新到优化器
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # 恢复上一次中断训练时所完成训练的轮数，接下来从这个轮数开始继续向后训练
    start_epoch = checkpoint['epoch']
    
    print('Resume from epoch {}'.format(args.c))
  
  else:
    print('Training from scratch!')
    
  # 开始进行模型训练
  # 注意：torch为我们封装了模型进入训练状态，进入验证状态的方法，开启训练前需要确保模型加入训练状态
  model.train()
  
  for epoch in range(start_epoch, HP.epochs):
    print('Starting Epoch:{}, Step:{}'.format(epoch, len(trainLoader) / HP.batch_size))
    
  # 依据一个batch_size读取数据
    for batch in trainLoader:
      
      # 加载数据
      x, y = batch
      
      # 梯度归零
      optimizer.zero_grad()
      
      # 前向传播
      pred = model(x)
      
      # 计算损失
      loss = criterion(pred, y)
      
      # 开始反向传播
      loss.backward()
      
      # 更新参数
      optimizer.step()
      
      # 开始记录一个batch_size训练完成后产生的日志数据
      logger.add_scalar('Loss/Train', loss, step)
  
  # 开始记录验证集的日志数据 - 逻辑是，每隔训练了多少步做一次验证集上的验证
      if not step % HP.verbose_step: # 表示达到了新的10步训练，此时需要进行一次验证集loss计算
        
        # 计算模型在验证集上的loss
        eval_loss = evaluate(model, valLoader, criterion)
        logger.add_scalar('Loss/Dev', eval_loss, step)
        
      # 保存模型 - 依据设定的save_step，即规定每训练完成200步进行一次模型保存
      if not step % HP.save_step: 
        model_path_name = 'model_{}_{}.pth'.format(epoch, step) # 保存的模型文件的名称
        
        # 完成模型保存
        save_checkpoint(model, epoch, optimizer, os.path.join('model', model_path_name))
      
      # step需要自增
      step += 1
      
      # 日志数据行缓存进行刷新
      logger.flush()
      print('Epoch : {} / {}, Step : {}, Train Loss : {}, Dev Loss : {}'.format(epoch, HP.epoches, step, loss.item(), eval_loss))
      
  # 关闭日志读写
  logger.close()
  
if __name__ == '__main__':
  train()