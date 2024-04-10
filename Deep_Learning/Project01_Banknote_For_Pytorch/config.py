'''
Author: Szl
Date: 2024-04-10 10:33:54
LastEditors: Szl
LastEditTime: 2024-04-10 15:05:34
Description: 编写config配置文件，配置所需的超参
'''

# Package and Modules import statements
# -----------------------

# codings
# -----------------------

class Hyperparameter(object):

  '''
    超参数配置对象
  '''

  # 第一部分参数：data - 数据集相关的超参数配置
  
  device = 'cpu' # 该参数表示所需的硬件算力类型，取值为cuda还是cpu
  data_dir = './data' # 数据集的根目录路径
  data_path = './data/data_banknote_authentication.txt' # 指定原始数据集的路径
  
  trainset_path = './data/train.txt' # 划分完毕的训练集数据
  devset_path = './data/dev.txt' # 划分完毕的验证集数据
  testset_path = './data/test.txt' # 划分完毕的测试集数据
  
  in_features = 4 # 输入特征维度，特征的数量n
  out_dims = 2 # 输出特征维度，最终输出结果的所有可能的类别
  seed = 1234 # 固定的随机种子
  
  # 第二部分参数：demol structure - 模型结构相关的超参数配置
  
  layer_list = [in_features, 64, 128, 64, out_dims] # 使用一个list组织整个网络的结构以及各个神经网络层的输入和输出维度

  # 第三部分参数：training - 训练相关的超参数配置
    
  batch_size = 64 # 一次训练时输入的数据样本个数 - 直接决定模型的收敛速度，其次会直接影响模型训练的效率，所以超参在真正的工程化开发中需要调参（过小会导致模型收敛效率较低，过大会导致现存溢出，终端训练） - 但目前该值给定一个小于128大于32的值即可
  
  init_lr = 0.001 # 初始学习率Learning_rate
  epoches = 100 # 训练的轮数
  verbose_step = 10 # 设置每训练完成10次打印一次训练输出
  save_step = 200 # 每训练200步进行一次模型保存
  
HP = Hyperparameter()

# run test UseCase if current modules in main
# -----------------------

if __name__ == '__main__':
  pass