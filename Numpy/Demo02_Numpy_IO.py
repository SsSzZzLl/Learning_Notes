'''
Author: Szl
Date: 2024-01-09 13:55:47
LastEditors: Szl
LastEditTime: 2024-01-09 14:47:24
Description: 
'''

import numpy as np

#description: 生成一个fake data
arr_x = np.random.randn(10)
print(arr_x)

print("-----------------------------------")

#description: 保存npy文件
#np.save('fake_data1.npy', arr_x)

print("-----------------------------------")

#description: 保存npz文件
#arr_y = np.arange(0 ,10 ,1)
#np.savez('fake_data2.npz', xarr = arr_x, yarr = arr_y)

print("-----------------------------------")

#description: 保存数据为txt文本文档
arr = np.random.randint(0, 1000, size = (10, 10))
#np.savetxt('fake_data3.txt', arr, delimiter=';')
np.savetxt('fake_data4.csv', arr, delimiter=',')

print("-----------------------------------")

#description: 从npy或npz文件中读取数据
#res = np.load('fake_data1.npy')
#print(res)

#res = np.load('fake_data2.npz')['xarr']
#print(res)

print("-----------------------------------")

#description: 从txt文档或csv文档读取数据



if __name__ == '__main__':
    pass
