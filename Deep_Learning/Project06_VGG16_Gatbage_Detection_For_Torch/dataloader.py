# Package and Modules import statements
# -----------------------

import os
import torch
import torchvision.transforms as transforms

from config import HP
from torchvision import datasets
from torch.utils.data import DataLoader


# codings
# -----------------------


'''
  导入数据
'''

data_dir  = HP.data_root

classes = os.listdir(data_dir)

# print(classes)

'''
  图片转换
'''

train_transforms = transforms.Compose([
  transforms.Resize([224, 224]), # 将输入图片resize成统一尺寸
  # transforms.RandomHorizontalFlip(), # 随机水平翻转
  transforms.ToTensor(), # 将PIL Image或numpy.ndarray转换为tensor，并归一化到[0,1]之间
  transforms.Normalize( # 标准化处理-->转换为标准正太分布（高斯分布），使模型更容易收敛
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
  ) # 其中 mean=[0.485, 0.456, 0.406]与std=[0.229 ,0.224, 0.225] 从数据集中随机抽样计算得到的。
])
 
test_transform = transforms.Compose([
  transforms.Resize([224, 224]), # 将输入图片resize成统一尺寸
  transforms.ToTensor(), # 将PIL Image或numpy.ndarray转换为tensor，并归一化到[0,1]之间
  transforms.Normalize( # 标准化处理-->转换为标准正太分布（高斯分布），使模型更容易收敛
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
  ) # 其中 mean=[0.485, 0.456, 0.406]与std=[0.229, 0.224, 0.225] 从数据集中随机抽样计算得到的。
])

total_data = datasets.ImageFolder("./data/Garbage classification/Garbage classification",transform = train_transforms)

# print(total_data)

# print(total_data.class_to_idx)

'''
  划分数据集
'''
train_size = int(0.8 * len(total_data))
test_size = len(total_data) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(total_data, [train_size, test_size])

# print(train_dataset, test_dataset)


train_dl = DataLoader(
  train_dataset,
  batch_size = HP.batch_size,
  shuffle = True,
  num_workers = 0
)

test_dl = torch.utils.data.DataLoader(
  test_dataset,
  batch_size = HP.batch_size,
  shuffle = True,
  num_workers = 0
)

for X, y in test_dl:
  print("Shape of X [N, C, H, W]: ", X.shape)
  print("Shape of y: ", y.shape, y.dtype)
  break


# run test UseCase if current modules in main
# -----------------------

if __name__ == '__main__':
  pass