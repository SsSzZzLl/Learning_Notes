'''
Author: Szl
Date: 2024-04-15 10:36:04
LastEditors: Szl
LastEditTime: 2024-04-15 10:46:52
Description: 
'''

# Package and Modules import statements
# -----------------------

import torch
from torch import nn
import torch.nn.functional as F

# codings
# -----------------------

# define a layer
layer = nn.Linear(in_features = 16, out_features = 5)

# fake data
x = torch.randn(size = (8, 16))

# forward cal
layer_output = layer(x)

# not active result
print(layer_output)
print('-----------------------')

# sigmoid
layer_output_sigmoid = F.sigmoid(layer_output)

print(layer_output_sigmoid)
print('-----------------------')

# relu
layer_output_relu = F.relu(layer_output)

print(layer_output_relu)
print('-----------------------')

# leaky relu
layer_output_leaky_relu = F.leaky_relu(layer_output)

print(layer_output_leaky_relu)
print('-----------------------')

# Mish
def Mish(x):
  return x * F.tanh(F.softplus(x))

layer_output_mish = Mish(layer_output)

print(layer_output_mish)
print('-----------------------')

# run test UseCase if current modules in main
# -----------------------

if __name__ == '__main__':
  pass