'''
Author: Szl
Date: 2024-01-09 14:27:54
LastEditors: Szl
LastEditTime: 2024-01-09 14:30:59
Description: 
'''

import numpy as np

#description: 准备两个数组
arr1 = np.random.randint(0, 1000, size=(10, 10))
arr2 = np.random.randint(0, 1000, size=(10, 10))
print(arr1)
print(arr2)

print("-----------------------------------")

#description: 数组间的矩阵运算
print(arr1 + arr2)

