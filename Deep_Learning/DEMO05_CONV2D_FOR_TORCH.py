'''
Author: Szl
Date: 2024-04-16 09:05:02
LastEditors: Szl
LastEditTime: 2024-04-16 09:30:55
Description: 
'''

# Package and Modules import statements
# -----------------------

from torch import nn
from PIL import Image # PIL工具提供一个加载图像的方法，该方法会返回一个n通道二维数据
from torchvision import transforms # 图像的数据增广工具



# codings
# -----------------------

# load image
image = Image.open('./lena.jpg')
transform = transforms.Compose(
  [transforms.ToTensor()]
)

x = transform(image)

# 增加一维,增加的一维作为读取图像后获得的tensor数组的第一维
x = x.unsqueeze(0) # x dim:[N, C, H, W] N-batch_size

batch_size, n_channels, height, width = x.size()
print("x size:{}", format(x.size()))

# params
n_out_channels = 32 # 卷积核的通道数 - 创建的卷积层的输出通道数
kernel_size = 11 # 卷积核的大小（形状） - 一般建议给定为奇数，不要给定为偶数

# 创建卷积网络层
conv2d_nn = nn.Conv2d(
  in_channels = n_channels, # 表示卷积层的输入维 - 注意：这里的输入维表示输入图像（上层卷积层输出）的通道数
  out_channels = n_out_channels, # 表示当前卷积层经由(kernel_size, n_out_channels)形状的卷积核运算之后的输出图像的通道数
  kernel_size = (kernel_size, kernel_size), 
  stride = 1,
  padding = (kernel_size // 2, kernel_size // 2) # padding一般给定为kernel_size // 2
)

x_out = conv2d_nn(x)
print(x_out.size()) # NCHW - 1, 32, ?, ?

# run test UseCase if current modules in main
# -----------------------

if __name__ == '__main__':
  pass