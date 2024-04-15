# Package and Modules import statements
# -----------------------

import torch
from torch import nn
from torch.nn import functional as F

# codings
# -----------------------

# fake data
ourput_from_pre_layer = torch.randn(size = (8, 224, 224, 16)) # NHWC

'''
  NHWC
  N: batch size
  H: height
  W: width
  C: channel
'''

# BN Normalization 纵向规范化
bn_nrom = nn.BatchNorm2d(num_features = 16)
norm_out = bn_nrom(ourput_from_pre_layer.permute(0, 3, 1, 2)) # permute方法可以基于原始数据维度之上，自定义该数据的全新的维度 NCHW
print(norm_out.shape)

# LN Normalization 横向规范化
ln_nrom = nn.LayerNorm([224, 224, 16])
norm_out = ln_nrom(ourput_from_pre_layer)
print(norm_out.shape)

# run test UseCase if current modules in main
# -----------------------

if __name__ == '__main__':
  pass