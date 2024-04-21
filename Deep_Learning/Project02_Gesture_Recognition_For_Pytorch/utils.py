# Package and Modules import statements
# -----------------------

import os
from PIL import Image

# codings
# -----------------------

# 工具函数：递归遍历指定目录下的所有子目录下的所有文件
def recursive_fetching(root, suffix = ['jpg', 'png']):
  
  '''
    递归遍历指定目录下的所有文件，并构造每个文件的路径，最终返回读取的所有文件的构造完毕的相对路径
    :param root: 指定读取的某根路径
    :param suffix: 备选的文件后缀格式
    :return: 读取到的所有文件的构造完毕的相对路径，list类型
  '''
  
  all_file_path = []
  
  def get_all_files(path):
    
    # 根据指定传入的某路径，获取该路径下的所有目录或文件
    all_file_list = os.listdir(path)
    
    # 遍历这个目录下的所有目录或文件
    for file in all_file_list:
      
      file_path = os.path.join(path, file)
      
      # 如果file_path是目录
      if os.path.isdir(file_path):
        
        # 递归调用get_all_files函数，继续对当前目录做递归调用
        get_all_files(file_path)
        
      # 否则，如果不是目录，是文件，则保存该文件的路径及文件名称
      elif os.path.isfile(file_path):
        
        all_file_path.append(file_path)

  get_all_files(root)
  
  return [it for it in all_file_path if os.path.split(it)[-1].split('.')[-1] in suffix]

# 工具函数：加载划分所得的某一个指定的数据集映射文件中记录的所有样本图像的相对路径及其所属类别(训练集，测试集，验证集)
def load_meta(meta_path):
  
  '''
    加载指定数据集
    :param meta_path: 划分所得的数据映射文件的路径
    :return: list类型对象
  '''

  with open(meta_path, 'r') as f:
    return [line.strip().split('|') for line in f.readlines()]

# 工具函数：根据读取的每一个图像样本的相对路径，加载该图像
def load_image(image_path):
  
  '''
    加载指定路径的图像
    :param image_path: str类型，表示指定图像的读取到的相对路径
    :return: 
  '''

  return Image.open(image_path)

# run test UseCase if current modules in main
# -----------------------

if __name__ == '__main__':
  pass