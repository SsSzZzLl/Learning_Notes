# FULL_CONNECT_NN

## DEMO01_FULL_CONNECT_NN_FOR_PYTORCH

```python
'''
Author: Szl
Date: 2024-04-09 11:42:05
LastEditors: Szl
LastEditTime: 2024-04-09 13:17:42
Description: 
'''

# Package and Modules import statements
# -----------------------

# import package
import torch
from torch import nn # 神经网络的子包
from torch.nn import functional as F # 神经网络的工具函数




# codings
# -----------------------

# create fake data

n_items = 10000 # samples
n_features = 2 # features

# global random state
torch.manual_seed(123)

# data
data_x = torch.randn(size = (n_items, n_features)).float()
# print(x)
y = torch.where(torch.subtract(data_x[:,0] * 0.5, data_x[:,1] * 1.5) + 0.02 > 0, 0, 1).long()
# print(y)

# 如果是神经网络求解二分类问题，建议对标签y进行独热编码 - one-hot编码
# one hot独热编码，将特征的取值按照不同的类别映射到一个维度为类别个数的高维空间，如果取到某类别的值，则高维空间下该类别所示的对应该样本的行位置处的值置为1，否则为0

'''
  sample shape : [10000, 2]
  target shape : [10000, 1]
  after target one-hot encode shape : [10000, 2]
  
  example :
    target = [
      1,
      0,
      1
    ]
    after one-hot target = [
      1, 0
      0, 1
      1, 0
    ]
'''

data_y = F.one_hot(y)
# print(y.size())

# 构建多层感知机模型
class BinaryClassificationModels(nn.Module):
  
  '''
    只要构建的是神经网络模型，则派生类必须继承nn.Module基类
  '''

  def __init__(self, in_features):
    
    # 派生类继承nn.Mudule之后，必须调用基类的初始化方法,完成派生类self参数绑定基类对象，随后派生类即可根据self自身对象之间调用基类的成员，包括属性和方法
    
    super().__init__()

    # muti perception 多层感知机
    # 创建一个神经网络的API是nn.linear(in_features表示输入维的维度, out_features表示输出维的维度, bias表示该神经网络是否需要偏置b)
    self.layer_1 = nn.Linear(in_features = in_features, out_features = 256, bias = True)
    
    # 如果是创建多个神经网络层，切记下一层的输入维保持与上一层的输出维一致
    self.layer_2 = nn.Linear(in_features = 256, out_features = 512, bias = True)
    self.layer_3 = nn.Linear(in_features = 512, out_features = 1024, bias = True)
    self.layer_4 = nn.Linear(in_features = 1024, out_features = 512, bias = True)
    self.layer_5 = nn.Linear(in_features = 512, out_features = 128, bias = True)
    self.layer_6 = nn.Linear(in_features = 128, out_features = 2, bias = True)
  
  # 前向计算 - 在pytorch中，单独封装为一个方法：forward - 注意：派生类的forward方法并不是派生类自行创建的，因为派生类继承了nn.Module基类，所以forward相当于是派生类在自身内部重写基类的forward方法！
  def forward(self, x):
     
    '''
      前向计算过程
      :param x: 传入的一条或者一个batch_size样本 - batch_size：训练时可以无需一次仅输入一条样本数据，torch允许模型训练时一次传入一批指定数量的数据进行训练，同时有助于应用小批量梯度下降  
    '''

    return F.sigmoid(self.layer_6(F.sigmoid(self.layer_5(F.sigmoid(self.layer_4(F.sigmoid(self.layer_3(F.sigmoid(self.layer_2(F.sigmoid(self.layer_1(x))))))))))))
  
# init params 
learning_rate = 0.001 # 学习率
epoches = 100 # 训练的轮数

# get model
model = BinaryClassificationModels(n_features)
  
# get optimizer：指定一个梯度下降优化器
opt = torch.optim.SGD(model.parameters(), lr = learning_rate) # 传参方式：传入需要优化的参数及指定的初始化学习率
  
# get loss : binaryEntropy - 二分类交叉熵损失函数
criteria = nn.BCELoss()

# build trainer

def trainer():
  
  # trainer main loop：训练的主循环 - 指定训练多少轮
  for epoch in range(epoches):
    
    steps = 0
    
    # 每一轮下指定输入一条样本进行训练
    for step in range(n_items):
      x = data_x[step]
      y = data_y[step] # 获取每次训练时所需的一条指定样本及其标签值
      
      # 训练第一步：首先使用优化器将上一步迭代时计算的梯度归零
      opt.zero_grad()
      
      # 训练第二步：完成一次前向计算获取模型的训练结果
      y_hat = model(x.unsqueeze(0)) # torch给定的前向计算的编程范式：基于神经网络类型所创建神经网络对象如果需要完成前向计算，无需显示手动调用forward方法，只需要将创建的模型对象继续做一次实例化，即可完成自动的前向计算过程，即对model再一次实例化调用，即可隐式触发派生类的forward方法调用
      
      # 训练第三步：根据损失函数开始梯度下降
      loss = criteria(y_hat, y.unsqueeze(0).float())

      # 训练第四步：反向传播，开启一次梯度下降过程，此过程会求解当前步的梯度，并随后完成一次参数更新
      loss.backward()
      
      # 参数更新
      opt.step()
      
      steps += 1
    
    if steps % 1000 == 0: # 测试输入，打印训练过程
      print('Epoch: {}, Step: {}, Loss: {}'.format(epoch, steps, loss.item()))
      
      
# run test UseCase if current modules in main
# -----------------------

if __name__ == '__main__':
  trainer()
```

