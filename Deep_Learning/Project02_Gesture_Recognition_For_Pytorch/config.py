'''
Author: Szl
Date: 2024-04-21 12:48:28
LastEditors: Szl
LastEditTime: 2024-04-21 13:29:26
Description: 完成基本的模型超参数配置
'''
# Package and Modules import statements
# -----------------------

# codings
# -----------------------

class Hyperparameters(object):

  '''
    超参数配置对象
  '''

  # data params
  device = 'cpu' # cuda - 表示使用gpu加速
  data_root = './data' # 指定当前工程下数据集的根路径
  
  # 准备四个文件的路径
  # 1.准备类别映射的json文件路径
  cls_mapper_path = './data/classes_mapper.json'
  
  # 2.准备划分完毕的训练集数据映射文件的路径
  metadata_train_path = './data/train_hand_gesture.txt'
  
  # 3.准备划分完毕的验证集数据映射文件的路径
  metadata_eval_path = './data/eval_hand_gesture.txt'
  
  # 4.准备划分完毕的测试集数据映射文件的路径
  metadata_test_path = './data/test_hand_gesture.txt'
  
  # 准备原始数据集的路径，方便后续直接读取完成数据集划分
  train_data_root = './data/shp_marcel_train/Marcel-Train'
  test_data_root = './data/shp_marcel_test/Marcel-Test'
  
  # model params
  
  # train params


# run test UseCase if current modules in main
# -----------------------

if __name__ == '__main__':
  pass