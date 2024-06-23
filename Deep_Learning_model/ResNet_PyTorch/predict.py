'''
Author: Szl
Date: 2024-06-16 23:12:41
LastEditors: Szl
LastEditTime: 2024-06-23 17:19:04
Description: 
'''

import os
import json
import torch
import matplotlib.pyplot as plt

from PIL import Image
from model import resnet34
from torchvision import transforms

def main():
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

  data_transform = transforms.Compose(
    [
      transforms.Resize(256),
      transforms.CenterCrop(224),
      transforms.ToTensor(),
      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]
  )
  
  img_path = './sunflower.jpg'
  assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
  img = Image.open(img_path)
  plt.imshow(img)
  img = data_transform(img)
  img = torch.unsqueeze(img, dim = 0)
  
  json_path = './class_indices.json'
  assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)
  
  with open(json_path, "r") as f:
    
    class_indict = json.load(f)
    
  # create model
  model = resnet34(num_classes=5).to(device)
  
  # load model weights
  weights_path = "./resNet34.pth"
  assert os.path.exists(weights_path), "file: '{}' dose not exist.".format(weights_path)
  model.load_state_dict(torch.load(weights_path, map_location = device))
  
  # prediction
  model.eval()
  with torch.no_grad():
    
    # predict class
    output = torch.squeeze(model(img.to(device))).cpu()
    predict = torch.softmax(output, dim=0)
    predict_cla = torch.argmax(predict).numpy()
    
  print_res = "class: {}   prob: {:.3}".format(class_indict[str(predict_cla)], predict[predict_cla].numpy())
  plt.title(print_res)
  
  for i in range(len(predict)):
    
    print("class: {:10}   prob: {:.3}".format(class_indict[str(i)], predict[i].numpy()))
   
  plt.show() 
  

if __name__ == '__main__':
  main()