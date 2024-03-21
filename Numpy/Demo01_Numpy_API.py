'''
Author: Szl
Date: 2024-01-09 10:24:50
LastEditors: Szl
LastEditTime: 2024-01-09 13:51:37
Description: 
'''

import numpy as np

lists = [i for i in range(10)]
print(lists)

new_ndarray = np.array(lists)
print(new_ndarray)

print("-----------------------------------")

#description: 创建全为1的数组
one_ndarray = np.ones(10)
print(one_ndarray)

print("-----------------------------------")

#description: 创建全为1的数组
zero_ndarray = np.zeros(10)
print(zero_ndarray)

print("-----------------------------------")

#description: 根据指定的数组形状和需要填充的值来创建一个ndarray对象
full_ndarray = np.full(shape=(3, 3), fill_value=2333)
print(full_ndarray)

print("-----------------------------------")

# description: 根据指定的起始位置、结束位置、步长来创建一个ndarray对象
arange_ndarray = np.arange(start=0, stop=80, step=2)
print(arange_ndarray)
print(type(arange_ndarray))

print("-----------------------------------")

#description: 构造一个具有等差数列的ndarray对象
fib_ndarray = np.linspace(start=0, stop=100, num=20)
print(fib_ndarray)
print(type(fib_ndarray))

print("-----------------------------------")

#description: 随机生成具有随机整数的ndarray对象
ramdon_ndarray = np.random.randint(low=1, high=1000, size=(10, 10))
print(ramdon_ndarray)

print("-----------------------------------")

#description: 随机生成具有随机浮点数的ndarray对象
ramdon_ndarray_float = np.random.random(size=(10, 10))
print(ramdon_ndarray_float)

print("-----------------------------------")

#description: 生成一个服从标准正态分布的n维随机变量构成的ndarray对象
gauss_ndarray = np.random.randn(10, 10)
print(gauss_ndarray)

print("-----------------------------------")

#description: 下述代码演示创建ndarray数组时为数据对象设置类型
float64_ndarray = np.array((1, 2, 3, 4, 5, 6, 7, 8, 9, 10), dtype = 'float64')
print(float64_ndarray)
print(type(float64_ndarray))

print("-----------------------------------")

#description: 在进行asarray转换时指定待存储数据的类型
as_ndarray = np.asarray((2, 22, 222, 2222), dtype = 'float32')
print(as_ndarray)
print(type(as_ndarray[1]))

print("-----------------------------------")

#description: 也可以先制定一种类型，先完成数组的创建，随后再单独更改数据的类型
as_ndarray_change = as_ndarray.astype('float64')
print(as_ndarray_change)
print(type(as_ndarray_change[1]))

print("-----------------------------------")

#description: 获取数组维度
print(ramdon_ndarray_float.ndim)

print("-----------------------------------")

#description: 获取数组形状
print(ramdon_ndarray_float.shape)

print("-----------------------------------")

#description: 获取数组中各元素对象的类型
print(ramdon_ndarray_float.dtype)

print("-----------------------------------")

#description: 获取数组中元素的总个数
print(ramdon_ndarray_float.size)

print("-----------------------------------")

#description: 获取数组中每一个元素占据内存的字节数大小 - ndarray数组中每一个元素占据的内存字节数大小是一致的
print(ramdon_ndarray_float.itemsize)
 
print("-----------------------------------")

#description: 获取数组的内部信息
print(ramdon_ndarray_float.flags)
 

if __name__ == '__main__':
    pass
