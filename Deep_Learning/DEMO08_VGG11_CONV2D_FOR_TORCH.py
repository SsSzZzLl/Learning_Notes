# Package and Modules import statements
# -----------------------

import torch
from torch import nn

# codings
# -----------------------

class VGG11(nn.Module):
  
  '''
    复现VGG11深度卷积神经网络
  '''
  
  def __init__(self, in_channels):
    super().__init__()
    
    # 网络结构的构建
    
    # 1.先构建卷积层部分
    self.conv2d_layers = nn.Sequential(
      
      # input layer dim:[N, 3, 224, 224]
      
      # 1.第一卷积层，输入通道数为in_channels，输出通道数out_channels为64，卷积核大小kernel_size为3*3，填充padding为kernel_size // 2
      nn.Conv2d(in_channels = in_channels, out_channels = 64, kernel_size = (3, 3), padding= (3 // 2, 3 // 2)), # [N, 64, 224, 224]
      nn.ReLU(), # 接ReLU激活
      nn.MaxPool2d(kernel_size = (2, 2), stride = 2), # 接非重叠最大池化 [N, 64, 112, 112]
      
      # 2.第二卷积层
      nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = (3, 3), padding= (3 // 2, 3 // 2)), # [N, 128, 112, 112]
      nn.ReLU(),
      nn.MaxPool2d(kernel_size = (2, 2), stride = 2), # [N, 128, 56, 56]
      
      # 3.第三卷积层
      nn.Conv2d(in_channels = 128, out_channels = 256, kernel_size = (3, 3), padding= (3 // 2, 3 // 2)), # [N, 256, 56, 56]
      nn.Conv2d(in_channels = 256, out_channels = 256, kernel_size = (3, 3), padding= (3 // 2, 3 // 2)), # [N, 256, 56, 56]
      nn.ReLU(),
      nn.MaxPool2d(kernel_size = (2, 2), stride = 2), # [N, 256, 28, 28]
      
      # 4.第四卷积层
      nn.Conv2d(in_channels = 256, out_channels = 512, kernel_size = (3, 3), padding= (3 // 2, 3 // 2)), # [N, 512, 28, 28]
      nn.Conv2d(in_channels = 512, out_channels = 512, kernel_size = (3, 3), padding= (3 // 2, 3 // 2)), # [N, 512, 28, 28]
      nn.ReLU(),
      nn.MaxPool2d(kernel_size = (2, 2), stride = 2), # [N, 512, 14, 14]
      
      # 5.第五卷积层
      nn.Conv2d(in_channels = 512, out_channels = 512, kernel_size = (3, 3), padding= (3 // 2, 3 // 2)), # [N, 512, 14, 14]
      nn.Conv2d(in_channels = 512, out_channels = 512, kernel_size = (3, 3), padding= (3 // 2, 3 // 2)), # [N, 512, 14, 14]
      nn.ReLU(),
      nn.MaxPool2d(kernel_size = (2, 2), stride = 2), # [N, 512, 7, 7]
    )
    
    # 2.再构建全连接层部分
    self.fc_layers = nn.Sequential(
      nn.Linear(in_features = 512 * 7 * 7, out_features = 4096),
      nn.ReLU(),
      
      nn.Linear(in_features = 4096, out_features = 4096),
      nn.ReLU(),
      
      nn.Linear(in_features = 4096, out_features = 1000)
    )
    
  def forward(self, x):
    out_from_conv2d = self.conv2d_layers(x)
    out_out_from_conv2d_flatten = out_from_conv2d.view(out_from_conv2d.size(0), -1) # 进入全连接层之前，将特征图展平 dim: 512 * 7 * 7
    return self.fc_layers(out_out_from_conv2d_flatten)
    
    
    

# run test UseCase if current modules in main
# -----------------------

if __name__ == '__main__':
  x = torch.randn(size = (8, 3, 224, 224))
  vgg11 = VGG11(in_channels = 3)
  output = vgg11(x)
  print('output size:{}'.format(output.size())) # [8, 1000]
  print(output)