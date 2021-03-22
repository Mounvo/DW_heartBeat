# Task3 特征工程

## 3.1 学习目标

+ 学习时间序列数据的特征预处理方法
+ 学习时间序列特征处理工具 Tsfresh（TimeSeires Fresh）的使用

## 3.2 内容介绍

+ 数据预处理

  时间序列数据格式处理

  加入时间步特征time

+ 特征工程

  时间序列特征构造

  特征筛选

  使用tsfresh进行时间序列特征处理

## 3.3 代码示例

### 3.3.1 导入包并读取数据

``` python
# 包导入
import pandas as pd
import numpy as np
import tsfresh as tsf
from tsfresh import extract_features, select_features
from tsfresh.utilities.dataframe_functions import impute
```

``` python
# 数据读取
data_train = pd.read_csv("data/train.csv")
data_test = pd.read_csv("data/testA.csv")
```

### 3.3.2 数据预处理

``` python
train_heartbeat_df = data_train["heartbeat_signals"].str.split(",", expand=True).stack()
train_heartbeat_df = train_heartbeat_df.reset_index()
train_heartbeat_df = train_heartbeat_df.set_index("level_0")
train_heartbeat_df.index.name = None
train_heartbeat_df.rename(columns={"level":"time", 0:"heartbeat_signals"}, inplace=True)
train_heartbeat_df["heartbeat_signals"] = train_heartbeat_df["heartbeat_signals"].astype(float)
train_heartbeat_df.head()
```

|      | time | heartbeat_signals |
| ---: | ---: | ----------------: |
|    0 |    0 |          0.991230 |
|    0 |    1 |          0.943533 |
|    0 |    2 |          0.764677 |
|    0 |    3 |          0.618571 |
|    0 |    4 |          0.379632 |

``` python
data_train_label = data_train["label"]
data_train = data_train.drop(["label","heartbeat_signals"], axis=1)
data_train = data_train.join(train_heartbeat_df)
data_train.head()
```

|      |   id | time | heartbeat_signals |
| ---: | ---: | ---: | ----------------- |
|    0 |    0 |    0 | 0.991230          |
|    0 |    0 |    1 | 0.943533          |
|    0 |    0 |    2 | 0.764677          |
|    0 |    0 |    3 | 0.618571          |
|    0 |    0 |    4 | 0.379632          |

``` python
data_train[data_train["id"]==0]
```

|      |   id | time | heartbeat_signals |
| ---: | ---: | ---: | ----------------: |
|    0 |    0 |    0 |          0.991230 |
|    0 |    0 |    1 |          0.943533 |
|    0 |    0 |    2 |          0.764677 |
|    0 |    0 |    3 |          0.618571 |
|    0 |    0 |    4 |          0.379632 |
|  ... |  ... |  ... |               ... |
|    0 |    0 |  200 |          0.000000 |
|    0 |    0 |  201 |          0.000000 |
|    0 |    0 |  202 |          0.000000 |
|    0 |    0 |  203 |          0.000000 |
|    0 |    0 |  204 |          0.000000 |

205 rows × 3 columns

因此得知，每个样本都由205个时间步的心电信号组成。

### 3.3.3 使用tsfresh进行时间序列特征处理

以下内容截止到写的时候还没有跑出来。。。结果就先不放上来了。

1、特征抽取`tsfresh`是一个Python第三方工具包。它可以自动计算大量的时间序列数据的特征。此外，该包还包含了特征重要性评估、特征选择的方法。

``` python
from tsfresh import extract_features

# 特征提取
train_features = extract_features(data_train, column_id='id', column_sort='time')
train_features
```

2、特征选择 train_features中包含了heartbeat_signals的779种常见的时间序列特征（所有这些特征的解释可以去看官方文档），这其中有的特征可能为NaN值（产生原因为当前数据不支持此类特征的计算），使用以下方式去除NaN值

``` python
from tsfresh.utilities.dataframe_functions import impute

# 去除抽取特征中的NaN值
impute(train_features)
```

接下来，按照特征和响应变量之间的相关性进行特征选择，这一过程包含两步：首先单独计算每个特征和响应变量之间的相关性，然后利用Benjamini-Yekutieli procedure [1] 进行特征选择，决定哪些特征可以被保留。

``` python
from tsfresh import select_features

# 按照特征和数据label之间的相关性进行特征选择
train_features_filtered = select_features(train_features, data_train_label)

train_features_filtered
```

