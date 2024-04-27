# Package and Modules import statements
# -----------------------

import os
from config import HP
from utils import recursive_fetching
import random
import json

# codings
# -----------------------

class_mappers = {
  'class2id' : {
    'biscuits' : 0,
    'bone' : 1,
    'bread' : 2,
    'plastic' : 3,
    'poop' : 4,
    'stone' : 5
  },
  'id2class' : {
    0 : 'biscuits',
    1 : 'bone',
    2 : 'bread',
    3 : 'plastic',
    4 : 'poop',
    5 : 'stone'
  }
}

def get_json():
  
  # 固定随机种子
  random.seed(HP.seed)
  json.dump(class_mappers, open(HP.cls_mapper_path, 'w'))
  
# 加载原始数据集，读取其中的所有样本图像
train_items = recursive_fetching(HP.train_data_root, ['jpg'])
test_items = recursive_fetching(HP.test_data_root, ['jpg'])

# 将原始数据集样本合并
dataset = train_items + test_items

# 获取原始样本总数是多少
dataset_num = len(dataset)

print('Total samples number: {}'.format(dataset_num))

# 随机重排 - 彻底打乱，随后开始数据集划分
random.shuffle(dataset)

# 准备一个类别到所有所属该类别样本的相对路径集合的字典映射
dataset_dicts = {}

# 遍历所有数据
for sample in dataset:
  cls_id = class_mappers['class2id'][os.path.split(sample)[-1].split('-')[0]] # 获取样本的所属类别且将其映射为一个数值
  
  # 如果当前类别不存在于dataset_dicts中
  if cls_id not in dataset_dicts:
    dataset_dicts[cls_id] = [sample] # 赋一个初值
  else:
    dataset_dicts[cls_id].append(sample) # 已经存在，则同类别的样本直接添加至列表
    
# 开始进行数据集划分
# 指定各个数据集的划分比例为8 : 1 : 1 
train_ratio, eval_ratio, test_ratio = 0.6, 0.3, 0.1

# 准备划分的数据集
train_set, eval_set, test_set = [], [], []

for cls_id, set_list in dataset_dicts.items():
  
  # 获取某类别下所有样本的数量
  length = len(set_list)
  
  # 获取每个类别下划分给训练集、验证集和测试集的样本数量
  train_num, eval_num = int(length * train_ratio), int(length * eval_ratio)
  test_num = length - train_num - eval_num
  
  # 再次随机重排
  random.shuffle(set_list)
  
  # 依据各个类别计算得出划分样本的数量依次截取指定数量的样本，完成本类别下所有样本的三个子集的划分
  train_set.extend(set_list[: train_num])
  eval_set.extend(set_list[train_num : train_num + eval_num])
  test_set.extend(set_list[train_num + eval_num :])
  
# 再次整体对划分的训练集、验证集和测试集进行随机重排
random.shuffle(train_set)
random.shuffle(eval_set)
random.shuffle(test_set)

print('训练集样本数量:{}, 验证集样本数量:{}, 测试集样本数量:{}'.format(len(train_set), len(eval_set), len(test_set)))
  
# 写入meta文件，进行保存
def write_meta(dataset_path, dataset):
  
  '''
    根据传入的指定的数据集完成指定数据集meta文件的写入保存
    :param dataset_path: meta文件的路径
    :param dataset: 指定的数据集
    :return: 
  '''

  with open(dataset_path, 'w') as f:
    for path in dataset:
      cls_id = class_mappers['class2id'][os.path.split(path)[-1].split('-')[0]]
      f.write('{}|{}\n'.format(cls_id, path))
  
for data_meta_path, dataset in zip([HP.metadata_train_path, HP.metadata_eval_path, HP.metadata_test_path], [train_set, eval_set, test_set]):
  write_meta(data_meta_path, dataset)
 

# run test UseCase if current modules in main
# -----------------------

if __name__ == '__main__':
  pass