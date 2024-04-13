'''
Author: Szl
Date: 2024-04-10 10:34:56
LastEditors: Szl
LastEditTime: 2024-04-13 17:36:21
Description: 
'''

# Package and Modules import statements
# -----------------------

import torch
from torch.utils.data import DataLoader
from dataset_banknote_dataloader import BankNoteDataset
from model import BankNokeClassificationModel
from config import HP


# codings
# -----------------------

# 完成模型推理，依据模型选型结束后选定的model_80_1200.pth模型进行推理脚本编写

# 准备一个新的模型
model = BankNokeClassificationModel()

checkpoint = torch.load('./model/model_80_1200.pth') # 加载模型选型后确定的保存下来的最优模型

# 加载模型参数
model.load_state_dict(checkpoint['model_state_dict'])
model.to(HP.device)

# 评估模型：使用accuracy_score准确率来评估模型

# 加载测试数据集
testset = BankNoteDataset(HP.testset_path)
test_loader = DataLoader(testset, batch_size = HP.batch_size, shuffle = True, drop_last = True)

# 模型加入评估验证状态
model.eval()

# 记录总测试样本数
total_cnt = 0

# 记录正确预测的样本数
correct_cnt = 0

with torch.no_grad():
  for batch in test_loader:
    x, y = batch
    pred = model(x)
    
    # 统计测试了多少条数据
    total_cnt += pred.size(0)
    
    # 统计正确分类了多少条样本
    correct_cnt += (torch.argmax(pred, 1) == y).sum() # 返回的是一行结果记录中概率最大得那个值得列索引 - 这个列索引（0-1）正好对应着预测得标签类别结果
    # 最终这里表示的是对于每一条样本，获取预测所属哪一类别得概率是最大的，即获取预测所属类别结果
    
# 计算准确率
accuracy_socre = correct_cnt / total_cnt

print('模型在测试集上的准确率为：{}'.format(accuracy_socre))

# run test UseCase if current modules in main
# -----------------------