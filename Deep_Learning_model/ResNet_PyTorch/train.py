'''
Author: Szl
Date: 2024-06-16 23:11:10
LastEditors: Szl
LastEditTime: 2024-06-23 15:13:53
Description: 
'''

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm
from model import resnet34
from torchvision import datasets, transforms

def main():
  
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  print('using {} device.'.format(device))
  
  data_transform = {
    'train': transforms.Compose(
      [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
      ]
    ),
    
    'val': transforms.Compose(
      [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
      ]
    )
  }