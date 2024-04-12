'''
Author: Szl
Date: 2024-04-10 10:34:14
LastEditors: Szl
LastEditTime: 2024-04-12 16:46:45
Description: 
'''

# Package and Modules import statements
# -----------------------

from torch import nn


# codings
# -----------------------

def get_loss_func():
  return nn.CrossEntropyLoss()


# run test UseCase if current modules in main
# -----------------------