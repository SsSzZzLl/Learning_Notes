'''
Author: Szl
Date: 2024-04-29 13:42:52
LastEditors: Szl
LastEditTime: 2024-04-29 15:57:37
Description: 
'''
# Package and Modules import statements
# -----------------------

import os
import torch
import torch.utils.data as Data

from config import HP
from PIL import Image
from torchvision import transforms as T

# codings
# -----------------------

gd_transform = T.Compose([
  T.Resize(HP.imageSize), # 保证所有输入图像尺寸一致
  T.RandomRotation(degrees = 45), # 图像旋转45度
  T.GaussianBlur(kernel_size = (3, 3)), # 为图像添加高斯噪声
  T.RandomHorizontalFlip(), # 对当前图像进行镜像操作
  T.ToTensor(), # 对图像做归一化操作，最终转化为各像素位取值为float32类型的tensor张量对象
  T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)) # 图像的标准化
])

def loadDataFromDir():
  
  '''
    从文件夹中获取数据
  '''
  
  images = [] # 图像列表
  labels = [] # 标签列表
  
  # 1. 获取根文件夹下所有分类文件夹
  for d in os.listdir(HP.data_root):
    
    # 2. 获取某一类型下所有的图片名称
    for imagePath in os.listdir(HP.data_root + d):
      
      # 3. 读取文件
      image = Image.open(HP.data_root + d + "/" + imagePath).convert('RGB')
      print("加载数据" + str(len(images)) + "条")
      
      # 4. 添加到图片列表中
      images.append(gd_transform(image))
      
      # 5. 构造label
      categoryIndex = HP.labels.index(d)  # 获取分类下标
      label = [0] * 8  # 初始化label
      label[categoryIndex] = 1 # 根据下标确定目标值
      label = torch.tensor(label,dtype = torch.float) # 转为tensor张量

      # 6. 添加到目标值列表
      labels.append(label)
      
      # 7. 关闭资源
      image.close() 

  return images, labels

def GarbageDataSet(Dataset):
  
  '''
    自定义DataSet
  '''

  def __init__(self):
    
    '''
    初始化DataSet
    '''
    
    images, labels = loadDataFromDir()  # 在文件夹中加载图片
    self.images = images
    self.labels = labels
    
  def __len__(self):
    
    '''
    返回数据总长度
    '''
    
    return len(self.images)

  def __getitem__(self, idx):
    
    image = self.images[idx]
    label = self.labels[idx]
    
    return image, label
  
def splitData(dataset):
  
  '''
    分割数据集
    :param dataset:
    :return:
  '''

  # 求解一下数据的总量
  total_length = len(dataset)
  
  # 确认一下将80%的数据作为训练集, 剩下的20%的数据作为测试集
  train_length = int(total_length * 0.8)
  validation_length = total_length - train_length
  
  # 利用Data.random_split()直接切分数据集, 按照80%, 20%的比例进行切分
  train_dataset,validation_dataset = Data.random_split(dataset = dataset, lengths = [train_length, validation_length])
  
  return train_dataset, validation_dataset

# 1. 分割数据集
train_dataset, validation_dataset = (GarbageDataSet()) 

# 2. 训练数据集加载器
trainLoader = GarbageDataSet(train_dataset, batch_size = HP.batch_size, shuffle = True, num_workers = HP.num_workers) 

# 3. 验证集数据加载器
valLoader = GarbageDataSet(validation_dataset, batch_size = HP.batch_size, shuffle = False,num_workers = HP.num_workers)
  
# run test UseCase if current modules in main
# -----------------------

if __name__ == '__main__':
  pass