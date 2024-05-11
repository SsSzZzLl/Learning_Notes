'''
Author: Szl
Date: 2024-04-29 13:42:39
LastEditors: Szl
LastEditTime: 2024-05-08 12:46:50
Description: 
'''
# Package and Modules import statements
# -----------------------
import time
import torch
import matplotlib.pyplot as plt

from torch import nn
from config import HP
from torch import optim
from models import model as GarbageDataSet
from dataloader import trainLoader, valLoader
from torch.utils.tensorboard import SummaryWriter
# codings
# -----------------------

# 1. 获取模型
model = GarbageDataSet
model.to(HP.device) # 模型传入cuda

# 2. 定义损失函数
criterion = nn.CrossEntropyLoss() # 交叉熵损失函数

# 3. 定义优化器
optimizer = optim.Adam(model.parameters(), lr = 0.001) # 自适应矩估计优化

# 4. 创建writer
writer = SummaryWriter(log_dir = HP.log_root, flush_secs = 500)

def train(epoch):
  
  '''
    训练函数
  '''
  
  # 1. 获取dataLoader
  loader = trainLoader
  
  # 2. 调整为训练状态
  model.train()
  
  print()
  print('========== Train Epoch:{} Start =========='.format(epoch))
  
  epochLoss = 0  # 每个epoch的损失
  epochAcc = 0  # 每个epoch的准确率
  correctNum = 0  # 正确预测的数量

  for data, label in loader:
    
    data, label = data.to(HP.device), label.to(HP.device)  # 加载到对应设备
    
    batchAcc = 0  # 单批次正确率
    batchCorrectNum = 0  # 单批次正确个数
    optimizer.zero_grad()  # 清空梯度
    output = model(data)  # 获取模型输出
    # label = label.float()
    label = torch.as_tensor(label, dtype=torch.long)
    
    # if len(output.shape) == 1: # 方式出现训练最后一个step时，出现v是一维的情况
    # output = torch.unsqueeze(output, 0)

    loss = criterion(output, label)  # 计算损失
    
    loss.backward()  # 反向传播梯度
    optimizer.step()  # 更新参数
    epochLoss += loss.item() * data.size(0)  # 计算损失之和
  
    label.shape
  
    # 计算正确预测的个数
    labels = torch.argmax(label, dim = 0)
    outputs = torch.argmax(output, dim = 1)
    
    for i in range(0, len(labels)):
      
      if labels[i] == outputs[i]:
        
        correctNum += 1
        batchCorrectNum += 1
        
    batchAcc = batchCorrectNum / data.size(0)
    
    print("Epoch:{}\t TrainBatchAcc:{}".format(epoch, batchAcc))
  
    epochLoss = epochLoss / len(trainLoader.dataset)  # 平均损失
    epochAcc = correctNum / len(trainLoader.dataset)  # 正确率
  
  print("Epoch:{}\t Loss:{} \t Acc:{}".format(epoch, epochLoss, epochAcc))
  
  writer.add_scalar("train_loss", epochLoss, epoch)  # 写入日志
  writer.add_scalar("train_acc", epochAcc, epoch)  # 写入日志
  
  return epochAcc


def val(epoch):
  
  '''
    验证函数
    :param epoch: 轮次
    :return:
  '''
  
  # 1. 获取dataLoader
  loader = valLoader
  
  # 2. 初始化损失、准确率列表
  valLoss = []
  valAcc = []
  
  # 3. 调整为验证状态
  model.eval()
  
  print()
  print('========== Val Epoch:{} Start =========='.format(epoch))
  
  epochLoss = 0  # 每个epoch的损失
  epochAcc = 0  # 每个epoch的准确率
  correctNum = 0  # 正确预测的数量

  with torch.no_grad():
    
    for data, label in loader:
      
      data, label = data.to(HP.device), label.to(HP.device)  # 加载到对应设备
      batchAcc = 0  # 单批次正确率
      batchCorrectNum = 0  # 单批次正确个数
      output = model(data)  # 获取模型输出
      loss = criterion(output, label)  # 计算损失
      epochLoss += loss.item() * data.size(0)  # 计算损失之和
      
      # 计算正确预测的个数
      labels = torch.argmax(label, dim=1)
      outputs = torch.argmax(output, dim=1)
      
      for i in range(0, len(labels)):
        
        if labels[i] == outputs[i]:
          
          correctNum += 1
          batchCorrectNum += 1
          
      batchAcc = batchCorrectNum / data.size(0)
      
      print("Epoch:{}\t ValBatchAcc:{}".format(epoch, batchAcc))

      epochLoss = epochLoss / len(valLoader.dataset)  # 平均损失
      epochAcc = correctNum / len(valLoader.dataset)  # 正确率
    
    print("Epoch:{}\t Loss:{} \t Acc:{}".format(epoch, epochLoss, epochAcc))
    
    writer.add_scalar("val_loss", epochLoss, epoch)  # 写入日志
    writer.add_scalar("val_acc", epochAcc, epoch)  # 写入日志

  return epochAcc



# run test UseCase if current modules in main
# -----------------------

if __name__ == '__main__':
  
    maxAcc = 0.75
    
    for epoch in range(1, HP.epochs + 1):
      
        trainAcc = train(epoch)
        valAcc = val(epoch)
        
        if valAcc > maxAcc:
          
            maxAcc = valAcc
            
            # 保存最大模型
            torch.save(model, HP.model_root + "weather-" + time.strftime('%Y-%m-%d-%H-%M-%S', time.gmtime()) + ".pth")
            
    # 保存模型
    torch.save(model,HP.model_root+"weather-"+time.strftime('%Y-%m-%d-%H-%M-%S',time.gmtime())+".pth")
