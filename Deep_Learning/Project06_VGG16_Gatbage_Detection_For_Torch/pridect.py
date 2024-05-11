'''
Author: Szl
Date: 2024-05-11 19:50:27
LastEditors: Szl
LastEditTime: 2024-05-11 19:58:53
Description: 
'''
# Package and Modules import statements
# -----------------------

import torch
import matplotlib.pyplot as plt


from config import HP
from PIL import Image
from models import model
from train import best_model, test
from dataloader import total_data, train_transforms, test_dl, loss_fn

# codings
# -----------------------


'''
  模型评估
'''

best_model.eval()
epoch_test_acc, epoch_test_loss = test(test_dl, best_model, loss_fn)

print(epoch_test_acc, epoch_test_loss)

'''
  指定图片进行预测
'''
classes = list(total_data.class_to_idx)
 
def predict_one_image(image_path, model, transform, classes):
    test_img = Image.open(image_path).convert('RGB')
    plt.imshow(test_img)       # 展示预测的图片
 
    test_img = transform(test_img)
    img = test_img.to(HP.device).unsqueeze(0)
 
    model.eval()
    output = model(img)
 
    _,pred = torch.max(output,1)
    pred_class = classes[pred]
    print(f'预测结果是：{pred_class}')
 
# 预测训练集中的某张照片
predict_one_image(image_path='...', model = model, transform = train_transforms, classes = classes)


# run test UseCase if current modules in main
# -----------------------

if __name__ == '__main__':
  pass