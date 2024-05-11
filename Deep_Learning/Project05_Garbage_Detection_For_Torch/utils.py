'''
Author: Szl
Date: 2024-04-10 10:35:20
LastEditors: Szl
LastEditTime: 2024-05-07 08:19:42
Description: 定义工具函数
'''

# Package and Modules import statements
# -----------------------

import torch


# codings
# -----------------------

# 函数一：定义验证集训练过程函数
def evaluate(model_, devloader, crit):
  
  '''
  实现训练过程中使用验证集评估模型的训练误差
  :param model_: 构建好的神经网络模型
  :param devloader: 验证集数据加载对象dataloader
  :param crit: 损失函数
  :return: 验证集上的loss下降值
  '''

  # 临时让模型从训练状态切换至验证状态
  model_.eval()
  
  sum_loss = 0.
  
  # 一次完整的验证过程同意要求处于不存在梯度更新的上下文环境
  with torch.no_grad():
    for batch in devloader:
      
      # 获取一个batch_size的验证集数据
      x, y = batch
      
      # 前向计算
      pred = model_(x)
      
      # 计算损失
      loss = crit(pred, y)
      
      # 计算总loss
      sum_loss += loss.item()
      
  model_.train()
  return sum_loss / len(devloader)    
      

# 函数二：定义模型保存函数
def save_checkpoint(model_, epoch_, optim, checkpoint_path):
  
  '''
  实现模型训练时依训练步保存模型
  :param model_: 需要保存的模型
  :param epoch_: 当前训练到达的轮数
  :param optim: 优化器，目的是从优化器中恢复训练的参数
  :param checkpoint_path: 从哪一个检查点开始保存，保存的模型的路径
  :return: None
  '''
  
  save_dict = {
    'epoch' : epoch_, # 记录的训练轮数
    'model_state_dict' : model_.state_dict(), # 模型的参数
    'optimizer_state_dict' : optim.state_dict() # 优化器需要更新的参数
  }
  
  # 完成模型保存
  torch.save(save_dict, checkpoint_path)
  

# run test UseCase if current modules in main
# -----------------------