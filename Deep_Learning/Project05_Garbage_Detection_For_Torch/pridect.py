'''
Author: Szl
Date: 2024-04-29 13:43:03
LastEditors: Szl
LastEditTime: 2024-05-06 13:39:25
Description: 
'''
# Package and Modules import statements
# -----------------------

import torch
import torchvision.transforms as transforms
from PIL import Image
from config import HP

# codings
# -----------------------

def pridect(imagePath, modelPath):
  
  '''
  预测函数
  :param imagePath: 图片路径
  :param modelPath: 模型路径
  :return:
  '''
  
  # 1. 读取图片
  image = Image.open(imagePath)
  
  # 2. 进行缩放
  image = image.resize(HP.imageSize)
  image.show()
  
  # 3. 加载模型
  model = torch.load(modelPath)
  model = model.to(HP.device)
  
  # 4. 转为tensor张量
  transform = transforms.ToTensor()
  x = transform(image)
  x = torch.unsqueeze(x, 0)  # 升维
  x = x.to(HP.device)
  
  # 5. 传入模型
  output = model(x)
  
  # 6. 使用argmax选出最有可能的结果
  output = torch.argmax(output)
  print("预测结果：",HP.labels[output.item()])

# run test UseCase if current modules in main
# -----------------------

if __name__ == '__main__':
  pridect("","./model/weather-2022-10-14-07-36-57.pth")