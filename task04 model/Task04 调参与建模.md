# Task04 调参与建模

## 4.1 相关模型原理介绍

### 4.1.1 逻辑回归模型

https://blog.csdn.net/han_xiaoyang/article/details/49123419

### 4.1.2 决策树模型

https://blog.csdn.net/c406495762/article/details/76262487

### 4.1.3 GBDT模型

https://zhuanlan.zhihu.com/p/45145899

### 4.1.4 XGBoost模型

https://blog.csdn.net/wuzhongqiang/article/details/104854890

### 4.1.5 LightGBM模型

之后的过程主要用的是lightGBM模型，所以在这里就多写一些。

首先，XGBoost和LightGBM都是GBDT算法的工具，由于XGBoost的特点导致它的空间消耗和时间消耗都比较大，不能满足在大型数据集上的应用，lightGBM在XGBoost算法上进行了优化。

- 基于Histogram的决策树算法。
- 单边梯度采样 Gradient-based One-Side Sampling(GOSS)：使用GOSS可以减少大量只具有小梯度的数据实例，这样在计算信息增益的时候只利用剩下的具有高梯度的数据就可以了，相比XGBoost遍历所有特征值节省了不少时间和空间上的开销。
- 互斥特征捆绑 Exclusive Feature Bundling(EFB)：使用EFB可以将许多互斥的特征绑定为一个特征，这样达到了降维的目的。
- 带深度限制的Leaf-wise的叶子生长策略：大多数GBDT工具使用低效的按层生长 (level-wise) 的决策树生长策略，因为它不加区分的对待同一层的叶子，带来了很多没必要的开销。实际上很多叶子的分裂增益较低，没必要进行搜索和分裂。LightGBM使用了带有深度限制的按叶子生长 (leaf-wise) 算法。
- 直接支持类别特征(Categorical Feature)
- 支持高效并行
- Cache命中率优化

写不完了，再写代码不用看了。（谁让我大晚上才开始弄。。。）

https://blog.csdn.net/wuzhongqiang/article/details/105350579

补充：https://zhuanlan.zhihu.com/p/99069186

### 4.1.6 Catboost模型

https://mp.weixin.qq.com/s/xloTLr5NJBgBspMQtxPoFA

### 4.1.7 时间序列模型(选学)

RNN：https://zhuanlan.zhihu.com/p/45289691

LSTM：https://zhuanlan.zhihu.com/p/83496936

### 4.1.8 推荐教材：

《机器学习》 https://book.douban.com/subject/26708119/

《统计学习方法》 https://book.douban.com/subject/10590856/

《面向机器学习的特征工程》 https://book.douban.com/subject/26826639/

《信用评分模型技术与应用》https://book.douban.com/subject/1488075/

《数据化风控》https://book.douban.com/subject/30282558/



## 4.2 模型对比与性能评估

### 4.2.1 逻辑回归

- 优点
  - 训练速度较快，分类的时候，计算量仅仅只和特征的数目相关；
  - 简单易理解，模型的可解释性非常好，从特征的权重可以看到不同的特征对最后结果的影响；
  - 适合二分类问题，不需要缩放输入特征；
  - 内存资源占用小，只需要存储各个维度的特征值；
- 缺点
  - **逻辑回归需要预先处理缺失值和异常值【可参考task3特征工程】；**
  - 不能用Logistic回归去解决非线性问题，因为Logistic的决策面是线性的；
  - 对多重共线性数据较为敏感，且很难处理数据不平衡的问题；
  - 准确率并不是很高，因为形式非常简单，很难去拟合数据的真实分布；

### 4.2.2 决策树模型

- 优点
  - 简单直观，生成的决策树可以可视化展示
  - **数据不需要预处理，不需要归一化，不需要处理缺失数据**
  - 既可以处理离散值，也可以处理连续值
- 缺点
  - 决策树算法非常容易过拟合，导致泛化能力不强（可进行适当的剪枝）
  - 采用的是贪心算法，容易得到局部最优解

### 4.2.3 集成模型集成方法（ensemble method）

通过组合多个学习器来完成学习任务，通过集成方法，可以将多个弱学习器组合成一个强分类器，因此集成学习的泛化能力一般比单一分类器要好。

集成方法主要包括Bagging和Boosting，Bagging和Boosting都是将已有的分类或回归算法通过一定方式组合起来，形成一个更加强大的分类。两种方法都是把若干个分类器整合为一个分类器的方法，只是整合的方式不一样，最终得到不一样的效果。常见的基于Bagging思想的集成模型有：随机森林，基于Boosting思想的集成模型有：Adaboost、GBDT、XgBoost、LightGBM等。

**Baggin和Boosting的区别总结如下：**

- **样本选择上：** Bagging方法的训练集是从原始集中有放回的选取，所以从原始集中选出的各轮训练集之间是独立的；而Boosting方法需要每一轮的训练集不变，只是训练集中每个样本在分类器中的权重发生变化。而权值是根据上一轮的分类结果进行调整
- **样例权重上：** Bagging方法使用均匀取样，所以每个样本的权重相等；而Boosting方法根据错误率不断调整样本的权值，错误率越大则权重越大；
- **预测函数上：** Bagging方法中所有预测函数的权重相等；而Boosting方法中每个弱分类器都有相应的权重，对于分类误差小的分类器会有更大的权重；
- **并行计算上：** Bagging方法中各个预测函数可以并行生成；而Boosting方法各个预测函数只能顺序生成，因为后一个模型参数需要前一轮模型的结果。

### 4.2.3 数据集划分方法

我们一般将数据集划分为训练集和测试集，也就是使用训练集训练模型，用测试集验证模型，通常选择测试误差最小的模型作为最好的模型。

另一种方式是将数据划分为训练集(train)，验证集(validation)，测试集(test)三部分，网上对这三类的定义差别比较大，主要是验证集和测试集，这里使用的一种说法是，在训练集上训练模型，在验证集上评估模型，一旦找到最佳参数，就在测试集上最后测试一次，以测试集上的误差作为泛化误差的近似值。

对于数据集的划分一般有三种方法：

+ **留出法（Holdout cross validation）**：按照一定固定的比例将数据划分为训练集、验证集、测试集。

+ **自助法（Bootstrap）**：有放回的从数据集中取出一个样本放入训练集中，重复进行m次。这样得到的训练集大小约为63.2%，测试集大小约为36.8%。但得到的错误率或许比较悲观。（data mining课的笔记）

+ **k 折交叉验证（k-fold cross validation）**：**k折交叉验证**通常将数据集D分为k份，其中k-1份作为训练集，剩余的一份作为测试集，这样就可以获得k组训练/测试集，可以进行k次训练与测试，最终返回的是k个测试结果的均值。交叉验证中数据集的划分依然是依据**分层采样**的方式来进行。

  对于交叉验证法，其k值的选取往往决定了评估结果的稳定性和保真性，**通常k值选取10。**

  当k=1的时候，我们称之为**留一法**

### 4.2.4 模型评价方法

在task01中有较详细的评估指标介绍，这里使用`abs-sum`作为模型评价标准。
$$
{abs-sum={ \left| {0.1-0} \right| }+{ \left| {0.7-1} \right| }+{ \left| {0.1-0} \right| }+{ \left| {0.1-0} \right| }=0.6}
$$


## 4.3 代码示例

### 4.3.1 导入包

``` python
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score

import os
import seaborn as sns
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")
```

### 4.3.2 读取数据

`reduce_mem_usage`通过把`int64/float64`类型的数值转换为`int(float)32/16/8`，以达到减小内存的目的，在数据集大的时候都可以使用。

``` python
def reduce_mem_usage(df):
    start_mem = df.memory_usage().sum() / 1024**2 
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))
    
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() / 1024**2 
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    
    return df
```

``` python
# 读取数据
data = pd.read_csv('data/train.csv')
# 简单预处理
data_list = []
for items in data.values:
    data_list.append([items[0]] + [float(i) for i in items[1].split(',')] + [items[2]])

data = pd.DataFrame(np.array(data_list))
data.columns = ['id'] + ['s_'+str(i) for i in range(len(data_list[0])-2)] + ['label']

data = reduce_mem_usage(data)
```

```
Memory usage of dataframe is 157.93 MB
Memory usage after optimization is: 39.67 MB
Decreased by 74.9%
```

### 4.3.3 简单建模

建模之前先分离出训练集

``` python
from sklearn.model_selection import KFold
# 分离数据集特征和分类，方便进行交叉验证
X_train = data.drop(['id','label'], axis=1)
y_train = data['label']

# 5折交叉验证
folds = 5
seed = 2021
kf = KFold(n_splits=folds, shuffle=True, random_state=seed)
```

自定义`f1-score`评价指标，在模型迭代中返回验证集`f1-score`变化情况。

``` python
def f1_score_vali(preds, data_vali):
    labels = data_vali.get_label()
    preds = np.argmax(preds.reshape(4, -1), axis=0)
    score_vali = f1_score(y_true=labels, y_pred=preds, average='macro')
    return 'f1_score', score_vali, True
```

使用Lightgbm进行建模

``` python
from sklearn.model_selection import train_test_split
import lightgbm as lgb
# 数据集划分
X_train_split, X_val, y_train_split, y_val = train_test_split(X_train, y_train, test_size=0.2)
train_matrix = lgb.Dataset(X_train_split, label=y_train_split)
valid_matrix = lgb.Dataset(X_val, label=y_val)

params = {
    "learning_rate": 0.1,
    "boosting": 'gbdt',   # 设置提升类型
    "lambda_l2": 0.1,  # L2正则化
    "max_depth": -1,
    "num_leaves": 128, # 一棵树上的叶子数
    "bagging_fraction": 0.8, # 在不进行重采样的情况下随机选择部分数据
    "feature_fraction": 0.8, # 在每棵树训练之前选择80%的参数
    "metric": None,
    "objective": "multiclass",  # 目标函数
    "num_class": 4,
    "nthread": 10,
    "verbose": -1,
}

# 训练模型
model = lgb.train(params, 
                  train_set=train_matrix, 
                  valid_sets=valid_matrix, 
                  num_boost_round=2000, 
                  verbose_eval=50, 
                  early_stopping_rounds=200,
                  feval=f1_score_vali)
```

（队长分享了LightGBM的官方文档，200多页，包括参数详情和调参思路之类的）

```
Training until validation scores don't improve for 200 rounds
[50]	valid_0's multi_logloss: 0.051365	valid_0's f1_score: 0.958297
[100]	valid_0's multi_logloss: 0.0467873	valid_0's f1_score: 0.966206
[150]	valid_0's multi_logloss: 0.0492876	valid_0's f1_score: 0.967733
[200]	valid_0's multi_logloss: 0.0515335	valid_0's f1_score: 0.968641
[250]	valid_0's multi_logloss: 0.0534033	valid_0's f1_score: 0.968494
Early stopping, best iteration is:
[90]	valid_0's multi_logloss: 0.0464533	valid_0's f1_score: 0.965314
```

在验证集上进行验证

``` python
val_pre_lgb = model.predict(X_val, num_iteration=model.best_iteration)
preds = np.argmax(val_pre_lgb, axis=1)
score = f1_score(y_true=y_val, y_pred=preds, average='macro')
print('未调参前lightgbm单模型在验证集上的f1：{}'.format(score))
```

``` python
未调参前lightgbm单模型在验证集上的f1：0.9653136456453539
```

``` python
# 使用五折交叉法划分数据集
cv_scores = []
for i, (train_index, valid_index) in enumerate(kf.split(X_train, y_train)):
    print('************************************ {} ************************************'.format(str(i+1)))
    X_train_split, y_train_split, X_val, y_val = X_train.iloc[train_index], y_train[train_index], X_train.iloc[valid_index], y_train[valid_index]
    
    train_matrix = lgb.Dataset(X_train_split, label=y_train_split)
    valid_matrix = lgb.Dataset(X_val, label=y_val)

    params = {
                "learning_rate": 0.1,
                "boosting": 'gbdt',  
                "lambda_l2": 0.1,
                "max_depth": -1,
                "num_leaves": 128,
                "bagging_fraction": 0.8,
                "feature_fraction": 0.8,
                "metric": None,
                "objective": "multiclass",
                "num_class": 4,
                "nthread": 10,
                "verbose": -1,
            }
    
    model = lgb.train(params, 
                      train_set=train_matrix, 
                      valid_sets=valid_matrix, 
                      num_boost_round=2000, 
                      verbose_eval=100, 
                      early_stopping_rounds=200,
                      feval=f1_score_vali)
    
    val_pred = model.predict(X_val, num_iteration=model.best_iteration)
    
    val_pred = np.argmax(val_pred, axis=1)
    cv_scores.append(f1_score(y_true=y_val, y_pred=val_pred, average='macro'))
    print(cv_scores)

print("lgb_scotrainre_list:{}".format(cv_scores))
print("lgb_score_mean:{}".format(np.mean(cv_scores)))
print("lgb_score_std:{}".format(np.std(cv_scores)))
```

``` 
lgb_scotrainre_list:[0.9674515729721614, 0.9656700872844327, 0.9700043639844769, 0.9655663272378014, 0.9631137190307674]
lgb_score_mean:0.9663612141019279
lgb_score_std:0.0022854824074775683
```

**笔记**

` KFold(n_splits=’warn’, shuffle=False, random_state=None)`

+ **参数** n_splits 表示划分为几块（至少是2；shuffle 表示是否打乱划分，默认False，即不打乱；random_state 表示是否固定随机起点，Used when shuffle == True.

+ **方法**  `split(X[,Y,groups])` 返回分类后数据集的index

+ ​         `get_n_splits([X, y, groups]) ` 返回分的块数

  例：

  ``` python
  for i, (train_index, valid_index) in enumerate(kf.split(X_train, y_train)):
      print(i,train_index,valid_index)
  ```

  ```
  0 [    0     1     4 ... 99995 99997 99999] [    2     3     9 ... 99985 99996 99998]
  1 [    0     1     2 ... 99997 99998 99999] [    7     8    10 ... 99979 99990 99995]
  2 [    1     2     3 ... 99995 99996 99998] [    0     6    17 ... 99992 99997 99999]
  3 [    0     2     3 ... 99997 99998 99999] [    1     5    11 ... 99982 99984 99988]
  4 [    0     1     2 ... 99997 99998 99999] [    4    26    29 ... 99981 99993 99994]
  ```



### 4.3.4 模型调参

只试了用贝叶斯调参，记录如下。

贝叶斯调参的基本思想就是，给定优化的目标函数(广义的函数，只需指定输入和输出即可，无需知道内部结构以及数学性质)，通过不断地添加样本点来更新目标函数的后验分布(高斯过程,直到后验分布基本贴合于真实分布）。简单的说，就是利用已有的先验信息去找到使目标函数达到全局最大的参数。

贝叶斯调参的步骤如下：

- 定义优化函数(rf_cv）
- 建立模型
- 定义待优化的参数
- 得到优化结果，并返回要优化的分数指标

``` python
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer # 这句不加会报错提示没有make_scorer
"""定义优化函数"""
def rf_cv_lgb(num_leaves, max_depth, bagging_fraction, feature_fraction, bagging_freq, min_data_in_leaf, 
              min_child_weight, min_split_gain, reg_lambda, reg_alpha):
    # 建立模型
    model_lgb = lgb.LGBMClassifier(boosting_type='gbdt', objective='multiclass', num_class=4,
                                   learning_rate=0.1, n_estimators=5000,
                                   num_leaves=int(num_leaves), max_depth=int(max_depth), 
                                   bagging_fraction=round(bagging_fraction, 2), feature_fraction=round(feature_fraction, 2),
                                   bagging_freq=int(bagging_freq), min_data_in_leaf=int(min_data_in_leaf),
                                   min_child_weight=min_child_weight, min_split_gain=min_split_gain,
                                   reg_lambda=reg_lambda, reg_alpha=reg_alpha,
                                   n_jobs= 8
                                  )
    f1 = make_scorer(f1_score, average='micro')
    val = cross_val_score(model_lgb, X_train_split, y_train_split, cv=5, scoring=f1).mean()

    return val
```

``` python
from bayes_opt import BayesianOptimization
"""定义优化参数"""
bayes_lgb = BayesianOptimization(
    rf_cv_lgb, 
    {
        'num_leaves':(10, 200),
        'max_depth':(3, 20),
        'bagging_fraction':(0.5, 1.0),
        'feature_fraction':(0.5, 1.0),
        'bagging_freq':(0, 100),
        'min_data_in_leaf':(10,100),
        'min_child_weight':(0, 10),
        'min_split_gain':(0.0, 1.0),
        'reg_alpha':(0.0, 10),
        'reg_lambda':(0.0, 10),
    }
)

"""开始优化"""
bayes_lgb.maximize(n_iter=10)
```

``` python
|   iter    |  target   | baggin... | baggin... | featur... | max_depth | min_ch... | min_da... | min_sp... | num_le... | reg_alpha | reg_la... |
|  1        |  0.9785   |  0.5174   |  10.78    |  0.8746   |  10.15    |  4.288    |  48.97    |  0.2337   |  42.83    |  6.551    |  9.015    |
|  2        |  0.9778   |  0.6777   |  41.77    |  0.5291   |  12.15    |  4.16     |  26.39    |  0.2461   |  55.78    |  6.528    |  0.6003   |
|  3        |  0.9745   |  0.5825   |  68.77    |  0.5932   |  8.36     |  9.296    |  77.74    |  0.7946   |  79.12    |  3.045    |  5.593    |
|  4        |  0.9802   |  0.9669   |  78.34    |  0.77     |  19.68    |  9.886    |  66.34    |  0.255    |  161.1    |  4.727    |  8.18     |
|  5        |  0.9836   |  0.9897   |  51.9     |  0.9737   |  16.82    |  2.001    |  42.1     |  0.03563  |  134.2    |  3.437    |  1.368    |
|  6        |  0.9749   |  0.5575   |  46.2     |  0.6518   |  15.9     |  7.817    |  34.12    |  0.341    |  153.2    |  7.144    |  7.899    |
|  7        |  0.9793   |  0.9644   |  55.08    |  0.9795   |  18.5     |  2.085    |  41.22    |  0.7031   |  129.9    |  3.369    |  2.717    |
|  8        |  0.9819   |  0.5926   |  58.23    |  0.6149   |  16.81    |  2.911    |  39.91    |  0.1699   |  137.3    |  2.685    |  2.891    |
|  9        |  0.983    |  0.7796   |  50.38    |  0.7261   |  17.87    |  3.499    |  37.59    |  0.1404   |  136.1    |  2.442    |  6.621    |
|  10       |  0.9843   |  0.638    |  49.32    |  0.9282   |  11.33    |  6.504    |  43.21    |  0.288    |  137.7    |  0.2083   |  6.966    |
|  11       |  0.9798   |  0.8196   |  47.05    |  0.5845   |  9.075    |  2.965    |  46.16    |  0.3984   |  131.6    |  3.634    |  2.601    |
|  12       |  0.9726   |  0.7688   |  37.57    |  0.9811   |  10.26    |  1.239    |  17.54    |  0.9651   |  46.5     |  8.834    |  6.276    |
|  13       |  0.9836   |  0.5214   |  48.3     |  0.8203   |  19.13    |  3.129    |  35.47    |  0.08455  |  138.2    |  2.345    |  9.691    |
|  14       |  0.9738   |  0.5617   |  45.75    |  0.8648   |  18.88    |  4.383    |  46.88    |  0.9315   |  141.8    |  4.968    |  5.563    |
|  15       |  0.9807   |  0.8046   |  47.05    |  0.6449   |  12.38    |  0.3744   |  41.13    |  0.6808   |  138.7    |  0.8521   |  9.461    |
=================================================================================================================================================
```

``` python
# 得到优化结果
bayes_lgb.max
```

```
{'target': 0.98515,
 'params': {'bagging_fraction': 0.5475828107557895,
  'bagging_freq': 15.617833530397629,
  'feature_fraction': 0.6961126413264633,
  'max_depth': 12.922325713793226,
  'min_child_weight': 6.074439532708633,
  'min_data_in_leaf': 42.767569531848046,
  'min_split_gain': 0.02235626827612547,
  'num_leaves': 164.94630233682383,
  'reg_alpha': 2.7790507436397647,
  'reg_lambda': 7.813417830605818}}
```

之后还需根据参数建立新的模型，以及在验证集上进行验证

``` python
# 调整一个较小的学习率，并通过cv函数确定当前最优的迭代次数
base_params_lgb = {
                    'boosting_type': 'gbdt',
                    'objective': 'multiclass',
                    'num_class': 4,
                    'learning_rate': 0.01,
                    'num_leaves': 138,
                    'max_depth': 11,
                    'min_data_in_leaf': 43,
                    'min_child_weight':6.5,
                    'bagging_fraction': 0.64,
                    'feature_fraction': 0.93,
                    'bagging_freq': 49,
                    'reg_lambda': 7,
                    'reg_alpha': 0.21,
                    'min_split_gain': 0.288,
                    'nthread': 10,
                    'verbose': -1,
}

cv_result_lgb = lgb.cv(
    train_set=train_matrix,
    early_stopping_rounds=1000, 
    num_boost_round=20000,
    nfold=5,
    stratified=True,
    shuffle=True,
    params=base_params_lgb,
    feval=f1_score_vali,
    seed=0
)
print('迭代次数{}'.format(len(cv_result_lgb['f1_score-mean'])))
print('最终模型的f1为{}'.format(max(cv_result_lgb['f1_score-mean'])))

```

``` 
迭代次数4833
最终模型的f1为0.961641452120875
```

``` python
import lightgbm as lgb
"""使用lightgbm 5折交叉验证进行建模预测"""
cv_scores = []
for i, (train_index, valid_index) in enumerate(kf.split(X_train, y_train)):
    print('************************************ {} ************************************'.format(str(i+1)))
    X_train_split, y_train_split, X_val, y_val = X_train.iloc[train_index], y_train[train_index], X_train.iloc[valid_index], y_train[valid_index]

    train_matrix = lgb.Dataset(X_train_split, label=y_train_split)
    valid_matrix = lgb.Dataset(X_val, label=y_val)

    params = {
                'boosting_type': 'gbdt',
                'objective': 'multiclass',
                'num_class': 4,
                'learning_rate': 0.01,
                'num_leaves': 138,
                'max_depth': 11,
                'min_data_in_leaf': 43,
                'min_child_weight':6.5,
                'bagging_fraction': 0.64,
                'feature_fraction': 0.93,
                'bagging_freq': 49,
                'reg_lambda': 7,
                'reg_alpha': 0.21,
                'min_split_gain': 0.288,
                'nthread': 10,
                'verbose': -1,
    }

    model = lgb.train(params, train_set=train_matrix, num_boost_round=4833, valid_sets=valid_matrix, 
                      verbose_eval=1000, early_stopping_rounds=200, feval=f1_score_vali)
    val_pred = model.predict(X_val, num_iteration=model.best_iteration)
    val_pred = np.argmax(val_pred, axis=1)
    cv_scores.append(f1_score(y_true=y_val, y_pred=val_pred, average='macro'))
    print(cv_scores)

print("lgb_scotrainre_list:{}".format(cv_scores))
print("lgb_score_mean:{}".format(np.mean(cv_scores)))
print("lgb_score_std:{}".format(np.std(cv_scores)))
```

``` 
lgb_scotrainre_list:[0.9615056903324599, 0.9597829114711733, 0.9644760387635415, 0.9622009947666585, 0.9607941521618003]
lgb_score_mean:0.9617519574991267
lgb_score_std:0.0015797109890455313
```

和之前的比较？？？

``` 
lgb_scotrainre_list:[0.9674515729721614, 0.9656700872844327, 0.9700043639844769, 0.9655663272378014, 0.9631137190307674]
lgb_score_mean:0.9663612141019279
lgb_score_std:0.0022854824074775683
```

