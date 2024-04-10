'''
Author: Szl
Date: 2024-04-10 10:35:20
LastEditors: Szl
LastEditTime: 2024-04-10 17:01:18
Description: 定义工具函数
'''

# Package and Modules import statements
# -----------------------



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

  pass

# 函数二：定义模型保存函数
def save_checkpoint(model_, epoch_, optim, checkpoint_path):
  
  '''
  实现模型训练时依训练步保存模型
  :param model_: 需要保存的模型
  :param epoch_: 当前训练到达的轮数
  :param optim: 优化器，目的是从优化器中恢复训练的参数
  :param checkpoint_path: 从哪一个检查点开始保存
  :return: None
  '''
  
  

# run test UseCase if current modules in main
# -----------------------