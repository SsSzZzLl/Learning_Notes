'''
Author: Szl
Date: 2024-03-20 11:24:53
LastEditors: Szl
LastEditTime: 2024-03-20 16:16:58
Description: 
'''


# -----------------------
# Package and Modules import statements
# -----------------------

# fromtorchvision import datasets of MNIST and transforms
from torchvision import datasets, transforms

# import base medules
import numpy as np

# import evluation metrics of accuracy_score
from sklearn.metrics import accuracy_score

import torch

from time import time, sleep

# -----------------------
# codings
# -----------------------

# download and load MNIST and dataset split for data and target
def load_and_split_dataset():
  # 1.download and load train dataset
  train_dataset = datasets.MNIST(
    root = './data', # Path
    download = True, # True:need to download
    transform = transforms.ToTensor(), # need change data type from ndarray to tensor 
    train = True # this is a train dataset
  )
  
  # 2.download and load test dataset
  test_dataset = datasets.MNIST(
    root = './data', # Path
    download = True, # True:need to download
    transform = transforms.ToTensor(), # need change data type from ndarray to tensor 
    train = False # this is a test dataset
  )

  # 2.split data and target from train dataset and test dataset
  train_x = [] # train data
  train_y = [] # train target
  
  # iter for each sample
  for i in range(len(train_dataset)): # i is index(row)
    
    # get sample and target
    images, target = test_dataset[i]

    # every sample append to list
    train_x.append(images.view(-1)) # images.view(-1) means reshape tensor
    train_y.append(target)
    
    # MNIST dataset 55000 train samples, we only need 5000 samples
    if i > 5000:
      break
  
  test_x = [] # test data
  test_y = [] # test target
  
  # iter for each sample
  for i in range(len(test_dataset)): # i is index(row)
    
    # get sample and target
    images, target = test_dataset[i]

    # every sample append to list
    test_x.append(images.view(-1)) # images.view(-1) means reshape tensor
    test_y.append(target)
    
    # MNIST dataset 10000 train samples, we only need 200 samples
    if i > 200:
      break

  print('simples all classes:{}'.format(set(train_y))) 
  return train_x, train_y, test_x, test_y

# define KNN model
def KNN_model(train_x, train_y, test_x, test_y, k):
  '''train_x - train dataset
     train_y - train target
     test_x - test dataset
     tesr_y - test target  
  '''
  
  # set local timestamp
  since = time()
  
  # get train dataset and test dataset of samples
  m = test_x.size(0) # == test_x.shape[0]
  n = train_x.size(0)
  
  # test dataset and train dataset original dimension(维度) is m * 1, ** 2 is square(平方) for each samples
  # sum(dim = 1 mean sum for line(samples),keepdim = True mean keep 2 dimension).expand(m, n) mean change the dimension for keep m * n
  xx = (test_x ** 2).sum(dim = 1, keepdim = True).expand(m, n) # a mapping test sample, b mapping train sample
  yy = (train_x ** 2).sum(dim = 1, keepdim = True).expand(n, m).transpose(0, 1) # mean reshape(转置)
  # xx and yy : a ^ 2 and b ^ 2
  print('xx shape:{}'.format(xx.size()))
  print('yy shape:{}'.format(yy.size()))
  
  # calculator(计算) number of k samples nearest neighor distance
  distance_matrix = xx + yy - 2 * test_x.matmul(train_x.transpose(0, 1)) # 矩阵乘法的API，因为test_x不是一个单独的标量值，是一个矩阵（数组）
  print('distance_matrix shape:{}'.format(distance_matrix.size()))
  
  # sort for distance_matrix, find which sample is the nearest neighbor 
  mink_idxs = distance_matrix.argsort(dim = -1) # index
  print('mink_idxs shape:{}'.format(mink_idxs.size()))
  
  # empty list to save k nums nearest neighbor
  res = [] # 承载的职责是：找到k个最近的邻居，且同时即代表模型的训练结束 - res表示预测结果
  
  for idxs in mink_idxs: # index
    # print('idxs:{}'.format(idxs.size()))
    res.append(np.bincount(np.array([train_y[idx] for idx in idxs[:k]])).argmax())
  print('res:{}'.format(res))
  
  # 断言调试，断定找到的k个最近邻的邻居个数与测试集样本是否一直
  assert len(res) == len(test_y)
  
  # cal accuracy_score
  print('accuracy_score:{}'.format(accuracy_score(test_y, res)))
  
  # cal training time
  time_end = time() - since
  
  print('KNN training time:{}m {}s'.format(time_end // 60, time_end % 60))
          
def test():
  train_x, train_y, test_x, test_y = load_and_split_dataset()
  KNN_model(torch.stack(train_x), train_y, torch.stack(test_x), test_y, 7)

# -----------------------
# run test UseCase if current modules in main
# -----------------------
if __name__ == '__main__':
  test()