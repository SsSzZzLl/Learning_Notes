'''
Author: Szl
Date: 2024-03-19 10:07:52
LastEditors: Szl
LastEditTime: 2024-03-19 11:33:51
Description:  
'''


# imort statements
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 解决坐标轴刻度负号乱码
plt.rcParams['axes.unicode_minus'] = False

# 解决中文乱码问题
plt.rcParams['font.sans-serif'] = ['Simhei']
plt.style.use('ggplot')

# 1.create fake data
row_data = {
    '颜色深度' : [14.13, 13.2, 13.16, 14.27, 13.24, 12.07, 12.43, 11.79, 12.37, 12.04],
    '酒精浓度' : [5.64, 4.28, 5.68, 4.80, 4.22, 2.76, 3.94, 3.10, 2.12, 2.6],
    '品种' : [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]  # 0代表“黑皮诺” 1代表“赤霞珠” labels
}

# create dataframe
wine_data = pd.DataFrame(
  data = row_data
)

print(wine_data)

# 2.拆分数据集，从中划分出特征和标签
x = np.array(wine_data.iloc[:, 0 : 2]) # 所有行全要，只要第一第二列
y = np.array(wine_data.iloc[:, -1]) # 所有行全要，只要最后一列


# create scatter
# 将labels = 1的品种的红酒，将其颜色深度作为散点图的x坐标值，酒精浓度作为散点图的y坐标值
# 将labels = 0的品种的红酒，将其颜色深度作为散点图的x坐标值，酒精浓度作为散点图的y坐标值
# 同时在准备一个测试数据，即给出一杯位置分类的红酒的特征值
# 观察散点图中各个样本点与未知测试数据的位置关系

new_data = np.array([12.03, 4.1]) # 未知分类的待标记样本

plt.scatter(
    x[y == 1, 0],
    x[y == 1, 1],
    color = 'red',
    label = '赤霞珠'
) # 画出标签y为1的，关于类别1：赤霞珠红酒的散点图

plt.scatter(
    x[y == 0, 0],
    x[y == 0, 1],
    color = 'black',
    label = '黑皮诺'
    
) # 画出标签y为0的，关于类别0：黑皮诺红酒的散点图

plt.scatter(
  new_data[0],
  new_data[1],
  color = 'yellow',
  label = '带标记测试样本'
)

# 设置散点图相关属性
plt.xlabel('酒精浓度')
plt.ylabel('颜色深度')
plt.legend(loc = 'lower right')
# 保存散点图
# plt.savefig('./wine_lable.png')

# 计算所有的已知类别类别的样本和待标记样本间的距离
# 使用欧氏距离计算公式 - 欧式距离如何实现：自己写
# 给出欧式距离计算公式，随后计算出所有样本点与待标记样本点之间的距离，获得距离数组
distance = [np.sqrt(np.sum((x_num - new_data) ** 2)) for x_num in x]
print(distance)

# 升序排序 - 距离最近的索引值
sort_list = np.argsort(distance)
print(sort_list)

# 假设此时指定k值为3，寻找前三个样本为new_data的最近的三个邻居
k = 3
# 从sort_list中找出前k个邻居样本
top_k = [y[i] for i in sort_list[0 : k]] # 获取前k个邻居样本各自所属的标签类别
print(top_k)

# 将top_k构造为series对象
print(pd.Series(top_k).value_counts()) # value_counts方法默认在展示样本所有各个类别的数量的同时，还会按照数量大小实现降序排列进行展示

class_map = {1 : '赤霞珠', 0 : '黑皮诺'}

print('带标记的测试样本：{}经由KNN最近邻算法，得出其所属的类别为：{}'.format(new_data, class_map[int(pd.Series(top_k).value_counts().index[0])]))


if __name__ == '__main__':
    pass