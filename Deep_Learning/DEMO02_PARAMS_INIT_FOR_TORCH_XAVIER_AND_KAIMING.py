'''
Author: Szl
Date: 2024-04-15 09:26:46
LastEditors: Szl
LastEditTime: 2024-04-15 10:36:21
Description: 
'''

# Package and Modules import statements
# -----------------------

from torch import nn


# codings
# -----------------------

# create a simlpe model

model = nn.Linear(in_features=15, out_features=256)

print(model.weight)

# Xavier 
nn.init.xavier_uniform_(model.weight, gain = nn.init.calculate_gain('tanh'))

print(model.weight)

# Kaiming
nn.init.kaiming_uniform_(model.weight, a = 1, mode = 'fan_in', nonlinearity = 'leaky_relu')

print(model.weight)


# run test UseCase if current modules in main
# -----------------------

if __name__ == '__main__':
  pass