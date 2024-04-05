'''
Author: Szl
Date: 2024-04-05 15:49:54
LastEditors: Szl
LastEditTime: 2024-04-05 20:30:46
Description: 
'''

# Package and Modules import statements
# -----------------------

# import modules
import torch
import torch.nn as nn # 神经网络的工具包 - 这里其实是在借助单层感知机来实现一个多元线性回归模型!
import numpy as np
import pandas as pd

from time import time, sleep

# import sklearn API
from sklearn.preprocessing import StandardScaler # 标准版操作 - 一次标准化的线性变化(x - μ) / σ (其中μ是均值，σ是方差)
from sklearn.model_selection import train_test_split

data_url = "http://lib.stat.cmu.edu/datasets/boston"
raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
target = raw_df.values[1::2, 2]

# codings
# -----------------------

# load data,and split data
class Preprocess_data(object):
  
  
  def __init__(self):
    
    self.x = data
    self.y = target
    self.preprocess_data()
    
        
  # preprocess data:standscaler
  def preprocess_data(self):
    
    # get standscaler obj and fit and transform
    self.x = StandardScaler().fit_transform(self.x)  
  
  
  # split data for train_set and test_set
  def split_data(self):
    
    # ndarray to tensor
    Xtrain, Xtest, Ytrain, Ytest = train_test_split(self.x, self.y, test_size = 0.3, random_state = 123)
    return torch.tensor(Xtrain, dtype=torch.float32), torch.tensor(Xtest, dtype=torch.float32), torch.tensor(Ytrain, dtype=torch.float32).view(-1, 1), torch.tensor(Ytest, dtype=torch.float32).view(-1, 1)

 
# get linear regressor model
class LinearRegressionModel(nn.Module):
  
  '''
    编程范式要求1
    借助torch提供的单层感知机（单层神经网络），不加入非线性激活，即可直接构建一个多元线性回归模型，只是当前回归模型的本质其实是一个单层非线性激活的神经网络
    所以：按照torch约束的编程范式，所有的自定义神经网络模型必须继承公共基类nn.Model - 因为由该基类所派生的所以派生类都是一个可调用对象
    什么叫可调用对象：该对象可以向方法、函数那样，通过添加一个括号可以直接将其调用，并在调用时传入参数，调用后返回结果对象
  '''


  def __init__(self, input_dim):
    
    '''
    description: 当前单层非线性激活感知机对象的初始化方法
    :param input_dim:指定样本的输入维大小 - 即指定数据集的特征数量
    '''    
    
    # 编程范式要点2：每一个继承自nn.model基类的派生类都需要在其派生类的初始化方法内地调用基类的初始化方法
    # 目的是为了派生类的对象可以随意访问基类的成员包含属性和方法
    super().__init__()
    
    # get a linear regressor
    self.linear = nn.Linear(input_dim, 1) # nn.Linear其实是在创建一个神经网络层 - 默认是一个全连接的神经网络层 - 但后后续的代码中不会对单程感知机做非线性映射所以实际上这就是多元线性回归模型
    
  # 现在多元线性回归模型已经构成，如何完成模型的预测？ - 在深度学习领域中，模型的一次预测称之为一次前向计算 - 模型的一次loss损失函数的梯度下降，称之为一次反向传播
  # 实现一个方法，完成一次前向计算
  def forward(self, x):
    '''
    description: cal forward
    :param x : a input sample
    '''    
    
    return self.linear(x)


# build train
def train(Xtrain, Xtest, Ytrain, Ytest, model, criterion, optimizer):
  
  '''
    linear regressor train
    :param Xtrain: train data
    :param Xtest: test data
    :param Ytrain: train target
    :param Ytest: test target
    :return: 
  '''
    
  # 构建模型训练过程
  
  # 指定训练的轮数epoches
  num_epoches = 1000
  for epoch in range(num_epoches):
    
    # 完成每一轮的训练过程
    
    # 1.先完成一次的前向计算 - 获取当前该条样本输入到模型后的预测结果
    outputs = model(Xtrain) # 前向计算的方法是否需要手动调用forward方法吗? 不用 - 因为当前模型类继承了nn.Model基类，该基类已经将其所有的派生类全部实现了callable对象(可调用对象)，也就是说现在只需要将该model视作为一个类，直接实例化，传入数据集即可自动完成前向计算，即可自动调用forward方法的执行

    # 2.需要对loss进行一次梯度下降的优化，即计算一次梯度下降后新的loss值
    loss = criterion(outputs, Ytrain)
    
    # 3.执行一次反向传播 - 梯度下降
    optimizer.zero_grad() # 梯度归零
    
    # 梯度下降
    loss.backward() # 一次反向传播
    optimizer.step() # 优化器更新一次参数
    
    # 每训练100步，打印一次训练日志 - 即输出当前已训练到多少轮，当前轮数下loss下降后达到的损失函数值得结果是多少
    if (epoch + 1) % 50 == 0:
      print('epoch:{},loss:{}'.format(epoch + 1, loss.item()))
      sleep(0.5)

def test():
  
  # get preprocess_data_obj 
  pre_data = Preprocess_data()
  
  # finish data split
  Xtrain, Xtest, Ytrain, Ytest = pre_data.split_data()
  
  # create linear model
  model = LinearRegressionModel(Xtrain.shape[1])
  
  # define optimize(优化器) - 因为线性回归模型是基于最小二乘法构建SSE残差平方和的极小化损失函数，这个过程损失函数需要经过梯度下降法到达局部最优解，二梯度下降的过程容易产生问题，需要使用优化器对齐梯度下降的迭代过程进行优化
  
  # loss function
  criterion = nn.MSELoss() # 直接选择基于均方误差作为评估指标来构建SSE损失函数 - 均方误差损失函数
  
  # create optimize
  optimizer = torch.optim.SGD(model.parameters(), lr = 0.01, ) # 随机梯度下降优化器 - lr : learning rate - 学习率，一次梯度下降过程中沿梯度方向下降的距离


  # training
  train(Xtrain, Xtest, Ytrain, Ytest, model, criterion, optimizer)
  
  # 开始测试集的测试 - 注意编程范式要点3：所有的前向计算必须在非反向传播环境下进行
  with torch.no_grad():
    
    predict = model(Xtest)
  
  # cal test loss
  test_loss = criterion(predict, Ytest)
  
  print('test loss:{}'.format(test_loss.item()))
  
    
# run test UseCase if current modules in main
# -----------------------

if __name__ == '__main__':
  test()