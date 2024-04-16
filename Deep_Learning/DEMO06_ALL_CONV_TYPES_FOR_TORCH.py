'''
Author: Szl
Date: 2024-04-16 10:28:00
LastEditors: Szl
LastEditTime: 2024-04-16 11:05:45
Description: 
'''
# Package and Modules import statements
# -----------------------

import torch
from torch import nn

# codings
# -----------------------

# def params number cal
def model_params_number_cal(model_):
  
  '''
    根据传入的卷积神经网络计算网络的总参数量
    :param model_: 卷积神经网络层
    :return: 该网络层的总参数量
  '''
  
  return sum([params.numel() for params in model_.parameters() if params.requires_grad])
    
# test1: full connected nn
model_fc = nn.Linear(
  in_features = 10 * 10 * 3,
  out_features = 10 * 10 * 30,
  bias = True
)

print('full connected nn params number is :{}'.format(model_params_number_cal(model_fc)))
print('-----------------------')

# test2: conv2d - 基础的卷积神经网络层
model_basic_conv2d = nn.Conv2d(
  in_channels = 3,
  out_channels = 30,
  kernel_size = (10, 10),
  bias = True
)

print('basic_conv2d params number is :{}'.format(model_params_number_cal(model_basic_conv2d)))
print('-----------------------')

# test3: dilated conv2d - 空洞卷积的卷积网络层
model_dilated_conv2d = nn.Conv2d(
  in_channels = 3,
  out_channels = 30,
  kernel_size = (10, 10),
  bias = True,
  dilation = (2, 2) # 表示每隔2格取一个像素点
)

print('dilated conv2d params number is :{}'.format(model_params_number_cal(model_dilated_conv2d)))
print('-----------------------')

# test4: group conv2d - 分组卷积的卷积网络层
model_group_conv2d = nn.Conv2d(
  in_channels = 3,
  out_channels = 30,
  kernel_size = (10, 10),
  bias = True,
  groups = 3 # 表示分组卷积，将输入通道分为3组
)

print('group conv2d params number is :{}'.format(model_params_number_cal(model_group_conv2d)))
print('-----------------------')

# test5: point-wise conv2d - 点卷积的卷积网络层
model_point_wise_conv2d = nn.Conv2d(
  in_channels = 3,
  out_channels = 30,
  kernel_size = (1, 1),
  bias = True
)

print('point-wise conv2d params number is :{}'.format(model_params_number_cal(model_point_wise_conv2d)))
print('-----------------------')

# test6: depth separable conv2d - 深度可分离卷积的卷积网络层
depth_conv2d = nn.Conv2d(
  in_channels = 3,
  out_channels = 3,
  kernel_size = (10, 10),
  groups = 3,
)
point_conv2d = nn.Conv2d(
  in_channels = 3,
  out_channels = 30,
  kernel_size = (1, 1)
)

print('depth separable conv2d params number is :{}'.format(model_params_number_cal(depth_conv2d) + model_params_number_cal(point_conv2d)))
print('-----------------------')

# test7: transpose conv2d - 反卷积/转置卷积的卷积网络层
transpose_conv2d = nn.ConvTranspose2d(
  in_channels = 3,
  out_channels = 30,
  kernel_size = (10, 10)
)
print('transpose conv2d params number is :{}'.format(model_params_number_cal(transpose_conv2d)))
print(transpose_conv2d(torch.randn(size = (1, 3, 10, 10))).size())
print('-----------------------')

# run test UseCase if current modules in main
# -----------------------

if __name__ == '__main__':
  pass