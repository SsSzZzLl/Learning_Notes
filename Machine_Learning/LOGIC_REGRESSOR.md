# LOGIC_REGRESSOR

## DEMO14_LOGIC_REGRESSOR_FOR_SKLEARN

```python
# 导包
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from IPython.core.interactiveshell import InteractiveShell # 这个对象设置所有行全部输出

# 导入鸢尾花数据集
from sklearn.datasets import load_iris

# 导入数据集划分工具
from sklearn.model_selection import train_test_split

# 导入逻辑回归模型
from sklearn.linear_model import LogisticRegression

# 导入分类模型评估指标accuracy_score
from sklearn.metrics import accuracy_score

# 设置该对象ast_node_interactivity属性的值为all，表示在notebook下每一行有输出的代码全部输出运算结果
InteractiveShell.ast_node_interactivity = "all"

# 解决坐标轴刻度负号乱码
plt.rcParams['axes.unicode_minus'] = False

# 解决中文乱码问题
plt.rcParams['font.sans-serif'] = ['Simhei']
plt.style.use('ggplot')
```

```python
dataset  = load_iris()
x = dataset.data
y = dataset.target
```

```python
# 数据集划分
Xtrain, Xtest, Ytrain, Ytest = train_test_split(x, y, test_size=0.3, random_state=123)
```

```python
# 构建逻辑回归模型
clf = LogisticRegression()

# 训练模型
clf = clf.fit(Xtrain, Ytrain)

# 模型前向计算（预测）
y_pred = clf.predict(Xtest)
y_pred
```

<img src="img/image-20240407155002953.png" alt="image-20240407155002953" style="zoom:50%;" />

```python
# 评估模型准确率
accuracy_score(y_pred, Ytest)
```

![image-20240407155027059](img/image-20240407155027059.png)

```python
# 获取预测结果的概率分布
clf.predict_proba(Xtest)
```

```python
# 查看截距
clf.intercept_

clf.coef_
```

<img src="img/image-20240407155053917.png" alt="image-20240407155053917" style="zoom:50%;" />

## 