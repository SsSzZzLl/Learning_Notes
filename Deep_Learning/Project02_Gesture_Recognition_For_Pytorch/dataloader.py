'''
Author: Szl
Date: 2024-04-21 12:51:52
LastEditors: Szl
<<<<<<< HEAD
LastEditTime: 2024-04-22 21:58:29
=======
LastEditTime: 2024-04-22 13:03:47
>>>>>>> f203aa3d913adecf074b0ffad7abbce0d28dd107
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

hg_transform = T.Compose([ # 完成一系列的数据增广操作
  T.Resize((112, 112)), # 保证所有输入图像尺寸一致
  T.RandomRotation(degrees = 45), # 图像旋转45度
  T.GaussianBlur(kernel_size = (3, 3)), # 为图像添加高斯噪声
  T.RandomHorizontalFlip(), # 对当前图像进行镜像操作
  T.ToTensor(), # 对图像做归一化操作，最终转化为各像素位取值为float32类型的tensor张量对象
  T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)) # 图像的标准化
])

# 构建数据集加载对象
class HandGestureDataset(torch.utils.data.Dataset):
  
  '''
    构建数据集加载对象
  '''

  def __init__(self, metadata_path):
    self.dataset = load_meta(meta_path = metadata_path)
    
  def __getitem__(self, idx):
    
    # 读取数据集中指定的某一个样本
    item = self.dataset[idx]
    
    # 获取该样本的路径和所属类别的映射值
    cls_id, path = int(item[0]), item[1]
    
    # 根据获取的样本图像的相对路径加载图像数据
    image = load_image(path)
    return hg_transform(image).to(HP.device), cls_id
  
  def __len__(self):
    return len(self.dataset)  

# run test UseCase if current modules in main
# -----------------------

if __name__ == '__main__':
  pass