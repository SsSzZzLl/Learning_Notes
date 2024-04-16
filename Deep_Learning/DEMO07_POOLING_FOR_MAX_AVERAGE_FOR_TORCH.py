'''
Author: Szl
Date: 2024-04-16 11:32:52
LastEditors: Szl
LastEditTime: 2024-04-16 11:45:05
Description: 
'''

# Package and Modules import statements
# -----------------------

from torch import nn
from PIL import Image
from torchvision import transforms

# codings
# -----------------------

transform = transforms.Compose([transforms.ToTensor()])

image = Image.open('./lena.jpg')

x = transform(image).unsqueeze(0)
batch_size, n_channels, height, width = x.size()
print('x size:{}'.format(x.size()))
print('-----------------------')

n_out_channels = 32
kernel_size = 11

conv2d_nn = nn.Conv2d(
  in_channels = n_channels,
  out_channels = n_out_channels,
  kernel_size = (kernel_size, kernel_size),
  stride = 1,
  padding = (kernel_size//2, kernel_size//2)
)

# 最大池化
pool_layer = nn.MaxPool2d(kernel_size = (2, 2), stride = 2) # 2 * 2 的池化子区域搭配步长为2，则表示此为非重叠最大池化
x_conv2d_out = conv2d_nn(x)
print('conv2d size:{}'.format(x_conv2d_out.size())) # 测试卷积层的输出尺寸
print('-----------------------')

pool_out = pool_layer(x_conv2d_out)
print('max pooling size:{}'.format(pool_out.size())) # 测试池化层的输出尺寸
print('-----------------------')

# 平均池化
pool_layer = nn.AvgPool2d(kernel_size = (2, 2), stride = 2)
x_conv2d_out = conv2d_nn(x)
print('conv2d size:{}'.format(x_conv2d_out.size())) 
print('-----------------------')

pool_out = pool_layer(x_conv2d_out)
print('average pooling size:{}'.format(pool_out.size())) 
print('-----------------------')










# run test UseCase if current modules in main
# -----------------------

if __name__ == '__main__':
  pass