'''
Author: Szl
Date: 2024-05-11 19:42:57
LastEditors: Szl
LastEditTime: 2024-05-11 19:49:07
Description: 
'''
# Package and Modules import statements
# -----------------------

import matplotlib.pyplot as plt

from config import HP
from train import train_acc, test_acc, train_loss, test_loss
# codings
# -----------------------

'''
  Loss和Accuracy图
'''

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号
plt.rcParams['figure.dpi'] = 100              # 分辨率
 
epochs_range = range(HP.epochs)
 
plt.figure(figsize = (12, 3)) # figsize 设置图形的大小，a 为图形的宽， b 为图形的高，单位为英寸
plt.subplot(1, 2, 1) # plt.subplot(nrows, ncols, index)，nrows：表示分割画布的行数，ncols：表示分割画布的列数，index：表示子图在画布中的索引，从左往右，从上往下编号
 
plt.plot(epochs_range, train_acc, label = 'Training Accuracy')
plt.plot(epochs_range, test_acc, label = 'Test Accuracy')
plt.legend(loc = 'lower right')
plt.title('Training and Validation Accuracy')
 
plt.subplot(1, 2, 2)
plt.plot(epochs_range, train_loss, label = 'Training Loss')
plt.plot(epochs_range, test_loss, label = 'Test Loss')
plt.legend(loc = 'upper right')
plt.title('Training and Validation Loss')
plt.savefig('./picture/loss_acc01.png')

# run test UseCase if current modules in main
# -----------------------

if __name__ == '__main__':
  pass