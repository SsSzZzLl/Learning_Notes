#  Package and Modules import statements
#  -----------------------

import copy
import torch
import torch.nn as nn

from config import HP
from model_VGG import model
from dataloader import train_dl, test_dl
#  codings
#  -----------------------

'''
  训练函数
'''
def train(dataloader, model, loss_fn, optimier):
  
  size = len(dataloader.dataset) # 训练集的大小
  num_batches = len(dataloader)

  train_loss, train_acc = 0, 0

  for X, y in dataloader:
    
    X, y = X.to(HP.device), y.to(HP.device)

    # 计算预测误差
    pred = model(X)
    loss = loss_fn(pred, y)
    
    # 反向传播
    optimier.zero_grad() # grad属性归零
    loss.backward() # 反向传播
    optimier.step() # 每一步自动更新
   
    # 记录acc和loss 
    train_acc += (pred.argmax(1) == y).type(torch.float).sum().item()
    train_loss += loss.item()

  train_acc /= size
  train_loss /= num_batches

  return train_acc, train_loss

'''
  测试函数
'''
def test(dataloader, model, loss_fn):
  size = len(dataloader.dataset) # 测试集的大小
  num_batches = len(dataloader)
  test_loss, test_acc = 0, 0
 
  # 当不进行训练时，停止梯度更新，节省计算内存消耗
  with torch.no_grad():
    
    for imgs, target in dataloader:
      
      imgs, target = imgs.to(HP.device), target.to(HP.device)

      # 计算loss
      target_pred = model(imgs)
      loss = loss_fn(target_pred, target)

      test_loss += loss.item()
      test_acc += (target_pred.argmax(1) == target).type(torch.float).sum().item()

  test_acc /= size
  test_loss /= num_batches

  return test_acc, test_loss


'''
  正式训练
'''
 
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
loss_fn = nn.CrossEntropyLoss()
 
epochs = 40
train_loss = []
train_acc = []
test_loss = []
test_acc = []
 
best_acc = 0        # 先设置一个最佳准确率，作为最佳模型的判别指标
 
for epoch in range(epochs):
  # adjust_learning_rate(optimizer, epoch, learn_rate)
  model.train()
  epoch_train_acc, epoch_train_loss = train(train_dl, model, loss_fn, optimizer)
  # scheduler.step()         # 更新学习率（调用官方动态学习率接口时使用）

  model.eval()
  epoch_test_acc, epoch_test_loss = test(test_dl, model, loss_fn)

  # 保存最佳模型到best_model
  if epoch_test_acc > best_acc:
      best_acc = epoch_test_acc
      best_model = copy.deepcopy(model)

  train_acc.append(epoch_train_acc)
  train_loss.append(epoch_train_loss)
  test_acc.append(epoch_test_acc)
  test_loss.append(epoch_test_loss)

  # 获取当前的学习率
  lr = optimizer.state_dict()['param_groups'][0]['lr']

  template = ('Epoch:{:2d}, Train_acc:{:.1f}%, Train_loss:{:.3f}, Test_acc:{:.1f}%, Test_loss:{:.3f}, lr:{:.2E}')
  print(template.format(epoch + 1, epoch_train_acc * 100, epoch_train_loss, epoch_test_acc * 100, epoch_test_loss,lr))

print(best_acc)

# 保存最佳模型到文件中
PATH = './model/best_model.pth' # 保存的参数文件名
torch.save(model.state_dict(), PATH)
 
print('Done')


#  run test UseCase if current modules in main
#  -----------------------

if __name__ == '__main__':
  pass