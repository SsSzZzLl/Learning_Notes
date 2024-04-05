# RIDGE_REGRESSOR

## DEMO12_RIDGE_REGRESSOR_FOR_SKLEARN

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 导入线性回归模型
from sklearn.linear_model import Ridge

# 导入数据集划分对象
from sklearn.model_selection import train_test_split

# 导入波士顿房价数据集
data_url = "http://lib.stat.cmu.edu/datasets/boston"
raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
target = raw_df.values[1::2, 2]

# 导入回归模型的评估指标
from sklearn.metrics import mean_squared_error, r2_score  

from IPython.core.interactiveshell import InteractiveShell # 这个对象设置所有行全部输出
  
# 设置该对象ast_node_interactivity的属性值为all，表示notebook下每一行有输出的代码全部输出运算结果
InteractiveShell.ast_node_interactivity = "all"

# 解决坐标轴刻度负号乱码
plt.rcParams['axes.unicode_minus'] = False

# 解决中文乱码问题
plt.rcParams['font.sans-serif'] = ['Simhei']
plt.style.use('ggplot')
```

```python
# 加载数据
x = data
y = target
x.shap
```

```python
# 数据集划分
Xtrain, Xtest, Ytrain, Ytest = train_test_split(x, y, test_size=0.3, random_state=123)
```

```python
reg_1 = Ridge(alpha = 0.001)

reg_1.fit(Xtrain, Ytrain)

# 查看回归系数
reg_1.coef_

# 查看截距
reg_1.intercept_
```

<img src="C:\Users\23820\AppData\Roaming\Typora\typora-user-images\image-20240405154412158.png" alt="image-20240405154412158" style="zoom:50%;" />

```python
# 导入带有k折交叉验证得岭回归模型
from sklearn.linear_model import RidgeCV
```

```python
reg_2 = RidgeCV(
  alphas = np.arange(1, 1001, 100),
  scoring = 'r2',
  store_cv_values = True # 是否保存每次验证结果
).fit(Xtrain, Ytrain)

# 没有交叉验证的岭回归模型的参数w的结果
reg_1.score(Xtest, Ytest)

# 带有10折交叉验证的岭回归模型的参数w的结果
reg_2.score(Xtest, Ytest)

# 查看所有10折交叉验证的结果 - 误差值
pd.DataFrame(reg_2.cv_values_)

# 10折交叉验证的测试误差的结果可以进一步求其均值
reg_2.cv_values_.mean(axis = 0)

# 可以选出经过k折交叉验证后所得的最佳正则化系数
reg_2.alpha_
```

<img src="C:\Users\23820\AppData\Roaming\Typora\typora-user-images\image-20240405154448852.png" alt="image-20240405154448852" style="zoom:50%;" />