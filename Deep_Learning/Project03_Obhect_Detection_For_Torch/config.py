'''
Author: Szl
Date: 2024-04-27 08:56:40
LastEditors: Szl
LastEditTime: 2024-04-27 10:31:13
Description: 
'''
# Package and Modules import statements
# -----------------------

import torch

# codings
# -----------------------

class Hyperparameters(object):
  
  # data params
  device = 'cuda' if torch.cuda.is_available() else 'cpu' # cuda - 表示使用gpu加速
  data_root = './data' # 指定当前工程下数据集的根路径

  # 1.准备类别映射的json文件路径
  cls_mapper_path = './data/classes_mapper.json'
  
  # 2.准备划分完毕的训练集数据映射文件的路径
  metadata_train_path = './data/train_object_detection.txt'
  
  # 3.准备划分完毕的验证集数据映射文件的路径
  metadata_eval_path = './data/eval_object_detection.txt'
  
  # 4.准备划分完毕的测试集数据映射文件的路径
  metadata_test_path = './data/test_object_detection.txt'
  
  # 准备原始数据集的路径，方便后续直接读取完成数据集划分
  train_data_root = './data/shp_marcel_train/Marcel-Train'
  test_data_root = './data/shp_marcel_test/Marcel-Test'
  
  # 类别的数量
  classes_num = 6
  
  # 固定随机种子，方便后续复现
  seed = 1234
  
  # model params
  # 输入图像的通道数
  data_channels = 3
  
  # 卷积核的形状，即kernel_size
  conv_kernel_size = 3
  
  # dropout正则化系数
  fc_drop_prob = 0.3
  
  # train params

  # batch_size - 一次训练输入多少条样本
  batch_size = 32
  
  # 初始化学习率
  init_lr = 5e-4
  
  # 训练轮数
  epochs = 100
  
  # 每隔多少步打印一次日志，保存一次日志
  verbose_step = 250
  
  # 每隔多少步保存一次模型
  save_model = 500

HP = Hyperparameters()

# run test UseCase if current modules in main
# -----------------------

if __name__ == '__main__':
  pass