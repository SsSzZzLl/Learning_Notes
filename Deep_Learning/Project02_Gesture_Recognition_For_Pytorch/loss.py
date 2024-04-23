# Package and Modules import statements
# -----------------------

import torch
from torch import nn
from torch.nn import functional as F

# codings
# -----------------------

def mish(x):
  
  '''
    声明mish激活函数
  '''

  return x * torch.tanh(F.softplus(x))

class Mish(nn.Module):
  
  '''
    将mish激活函数封装为torch可用的激活函数对象
  '''

  def __init__(self):
    super().__init__()
    
  def forward(self, x):
    return mish(x)
  
  
# run test UseCase if current modules in main
# -----------------------

if __name__ == '__main__':
  pass