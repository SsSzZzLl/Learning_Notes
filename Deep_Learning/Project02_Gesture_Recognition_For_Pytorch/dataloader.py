'''
Author: Szl
Date: 2024-04-21 12:51:52
LastEditors: Szl
LastEditTime: 2024-04-21 22:30:37
Description: 
'''
# Package and Modules import statements
# -----------------------

import torch
from torch.utils.data import DataLoader
from config import HP
from utils import recursive_fetching, load_meta, load_image
from torchvision import transforms as T

# codings
# -----------------------

T.Compose([ # 完成一系列的数据增广操作
  T.Resize(112, 112), # 保证所有输入图像尺寸一致
  
])


# run test UseCase if current modules in main
# -----------------------

if __name__ == '__main__':
  pass