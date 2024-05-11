'''
Author: Szl
Date: 2024-05-11 17:25:38
LastEditors: Szl
LastEditTime: 2024-05-11 17:28:02
Description: 
'''
# Package and Modules import statements
# -----------------------

import torch
import time

# codings
# -----------------------

class Hyperparameters(object):
  
  device = 'cuda' if torch.cuda.is_available() else 'cpu' # 设备配置 cuda是否可用，否则使用cpu计算
  
  data_root = './data/Garbage classification/Garbage classification/' # 图片文件的路径

  imageSize = (224,224) # 图片大小，在ResNet中，图片大小必须为224x224
  
  labels = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash'] # 标签名称 / 文件夹名称

  num_workers = 0 # 加载数据时使用的线程数量

  batch_size = 32 # 每次训练的样本数量
  
  init_lr = 5e-4 # 初始学习率 科学计数法表示
  
  epochs = 40 # 训练轮数为40轮
  
  log_root = "./log/" + time.strftime('%Y-%m-%d-%H-%M-%S',time.gmtime()) # 日志存放位置

  seed = 1234

  model_root = './model/' # 模型存放位置

HP = Hyperparameters()

# run test UseCase if current modules in main
# -----------------------

if __name__ == '__main__':
  pass