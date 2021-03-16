# Task1 赛题理解

## 

Tip:本次新人赛是Datawhale与天池联合发起的零基础入门系列赛事第五场 —— 零基础入门心电图心跳信号多分类预测挑战赛。

2016年6月，国务院办公厅印发《国务院办公厅关于促进和规范健康医疗大数据应用发展的指导意见》,文件指出健康医疗大数据应用发展将带来健康医疗模式的深刻变化，有利于提升健康医疗服务效率和质量。

赛题以心电图数据为背景，要求选手根据心电图感应数据预测心跳信号，其中心跳信号对应正常病例以及受不同心律不齐和心肌梗塞影响的病例，这是一个多分类的问题。通过这道赛题来引导大家了解医疗大数据的应用，帮助竞赛新人进行自我练习、自我提高。

比赛地址：https://tianchi.aliyun.com/competition/entrance/531883/introduction

### 1.1学习目标

- 理解赛题数据和目标，清楚评分体系。
- 完成相应报名，下载数据和结果提交打卡（可提交示例结果），熟悉比赛流程

### 1.2了解赛题

- 赛题概况
- 数据概况
- 预测指标
- 分析赛题

#### 1.2.1赛题概况

比赛要求参赛选手根据给定的数据集，建立模型，预测不同的心跳信号。赛题以预测心电图心跳信号类别为任务，数据集报名后可见并可下载，该该数据来自某平台心电图数据记录，总数据量超过20万，主要为1列心跳信号序列数据，其中每个样本的信号序列采样频次一致，长度相等。为了保证比赛的公平性，将会从中抽取10万条作为训练集，2万条作为测试集A，2万条作为测试集B，同时会对心跳信号类别（label）信息进行脱敏。

通过这道赛题来引导大家走进医疗大数据的世界，主要针对于于竞赛新人进行自我练习，自我提高。

#### 1.2.2数据概况

一般而言，对于数据在比赛界面都有对应的数据概况介绍（匿名特征除外），说明列的性质特征。了解列的性质会有助于我们对于数据的理解和后续分析。

Tip:匿名特征，就是未告知数据列所属的性质的特征列。

train.csv

- id 为心跳信号分配的唯一标识
- heartbeat_signals 心跳信号序列(数据之间采用“,”进行分隔)
- label 心跳信号类别（0、1、2、3）

testA.csv

- id 心跳信号分配的唯一标识
- heartbeat_signals 心跳信号序列(数据之间采用“,”进行分隔)

#### 1.2.3预测指标

选手需提交4种不同心跳信号预测的概率，选手提交结果与实际心跳类型结果进行对比，求预测的概率与真实值差值的绝对值。

具体计算公式如下：

总共有n个病例，针对某一个信号，若真实值为[y1,y2,y3,y4],模型预测概率值为[a1,a2,a3,a4],那么该模型的评价指标abs-sum为 
$$
{abs-sum={\mathop{ \sum }\limits_{{j=1}}^{{n}}{{\mathop{ \sum }\limits_{{i=1}}^{{4}}{{ \left| {y\mathop{{}}\nolimits_{{i}}-a\mathop{{}}\nolimits_{{i}}} \right| }}}}}}
$$
例如，某心跳信号类别为1，通过编码转成[0,1,0,0]，预测不同心跳信号概率为[0.1,0.7,0.1,0.1]，那么这个信号预测结果的abs-sum为
$$
{abs-sum={ \left| {0.1-0} \right| }+{ \left| {0.7-1} \right| }+{ \left| {0.1-0} \right| }+{ \left| {0.1-0} \right| }=0.6}
$$
多分类算法常见的评估指标如下：

其实多分类的评价指标的计算方式与二分类完全一样，只不过我们计算的是针对于每一类来说的召回率、精确度、准确率和 F1分数。

1、混淆矩阵（Confuse Matrix）

- （1）若一个实例是正类，并且被预测为正类，即为真正类TP(True Positive )
- （2）若一个实例是正类，但是被预测为负类，即为假负类FN(False Negative )
- （3）若一个实例是负类，但是被预测为正类，即为假正类FP(False Positive )
- （4）若一个实例是负类，并且被预测为负类，即为真负类TN(True Negative ）

第一个字母T/F，表示预测的正确与否；第二个字母P/N，表示预测的结果为正例或者负例。如TP就表示预测对了，预测的结果是正例，那它的意思就是把正例预测为了正例。

2.准确率（Accuracy） 准确率是常用的一个评价指标，但是不适合样本不均衡的情况，医疗数据大部分都是样本不均衡数据。 
$$
Accuracy=\frac{Correct}{Total}\ Accuracy = \frac{TP + TN}{TP + TN + FP + FN}
$$
 3、精确率（Precision）也叫查准率简写为P

**精确率(Precision)是针对预测结果而言的，其含义是在被所有预测为正的样本中实际为正样本的概率**在被所有预测为正的样本中实际为正样本的概率，精确率和准确率看上去有些类似，但是是两个完全不同的概念。精确率代表对正样本结果中的预测准确程度，准确率则代表整体的预测准确程度，包括正样本和负样本。
$$
Precision = \frac{TP}{TP + FP}
$$
 4.召回率（Recall） 也叫查全率 简写为R

**召回率(Recall)是针对原样本而言的，其含义是在实际为正的样本中被预测为正样本的概率**。 
$$
Recall = \frac{TP}{TP + FN}
$$
下面我们通过一个简单例子来看看精确率和召回率。假设一共有10篇文章，里面4篇是你要找的。根据你的算法模型，你找到了5篇，但实际上在这5篇之中，只有3篇是你真正要找的。

那么算法的精确率是3/5=60%，也就是你找的这5篇，有3篇是真正对的。算法的召回率是3/4=75%，也就是需要找的4篇文章，你找到了其中三篇。以精确率还是以召回率作为评价指标，需要根据具体问题而定。

5.宏查准率（macro-P）

计算每个样本的精确率然后求平均值 
$$
{macroP=\frac{{1}}{{n}}{\mathop{ \sum }\limits_{{1}}^{{n}}{p\mathop{{}}\nolimits_{{i}}}}}
$$
6.宏查全率（macro-R）

计算每个样本的召回率然后求平均值 
$$
 {macroR=\frac{{1}}{{n}}{\mathop{ \sum }\limits_{{1}}^{{n}}{R\mathop{{}}\nolimits_{{i}}}}} $$ 7.宏F1（macro-F1） $$ {macroF1=\frac{{2 \times macroP \times macroR}}{{macroP+macroR}}}
$$
与上面的宏不同，微查准查全，先将多个混淆矩阵的TP,FP,TN,FN对应位置求平均，然后按照P和R的公式求得micro-P和micro-R，最后根据micro-P和micro-R求得micro-F1

8.微查准率（micro-P）
$$
{microP=\frac{{\overline{TP}}}{{\overline{TP} \times \overline{FP}}}}
$$
 

 9.微查全率（micro-R）
$$
{microR=\frac{{\overline{TP}}}{{\overline{TP} \times \overline{FN}}}}
$$
 10.微F1（micro-F1） 
$$
{microF1=\frac{{2 \times microP\times microR }}{{microP+microR}}}
$$
 

## 2 baseline

```python
import os
import gc
import math

import pandas as pd
import numpy as np

import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
from sklearn.linear_model import SGDRegressor, LinearRegression, Ridge
from sklearn.preprocessing import MinMaxScaler


from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

from tqdm import tqdm
import matplotlib.pyplot as plt
import time
import warnings
warnings.filterwarnings('ignore')
```

### 读取数据

``` python
train = pd.read_csv('train.csv')
test=pd.read_csv('testA.csv')
train.head()
```

|      | id   | heartbeat_signals                                 | label |
| :--- | :--- | :------------------------------------------------ | :---- |
| 0    | 0    | 0.9912297987616655,0.9435330436439665,0.764677... | 0.0   |
| 1    | 1    | 0.9714822034884503,0.9289687459588268,0.572932... | 0.0   |
| 2    | 2    | 1.0,0.9591487564065292,0.7013782792997189,0.23... | 2.0   |
| 3    | 3    | 0.9757952826275774,0.9340884687738161,0.659636... | 0.0   |
| 4    | 4    | 0.0,0.055816398940721094,0.26129357194994196,0... | 2.0   |

``` python
test.head()
```

| id   | heartbeat_signals |                                                   |
| :--- | :---------------- | ------------------------------------------------- |
| 0    | 100000            | 0.9915713654170097,1.0,0.6318163407681274,0.13... |
| 1    | 100001            | 0.6075533139615096,0.5417083883163654,0.340694... |
| 2    | 100002            | 0.9752726292239277,0.6710965234906665,0.686758... |
| 3    | 100003            | 0.9956348033996116,0.9170249621481004,0.521096... |
| 4    | 100004            | 1.0,0.8879490481178918,0.745564725322326,0.531... |

### 数据预处理

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
# 简单预处理
train_list = []

for items in train.values:
    train_list.append([items[0]] + [float(i) for i in items[1].split(',')] + [items[2]])

train = pd.DataFrame(np.array(train_list))
train.columns = ['id'] + ['s_'+str(i) for i in range(len(train_list[0])-2)] + ['label']
train = reduce_mem_usage(train)

test_list=[]
for items in test.values:
    test_list.append([items[0]] + [float(i) for i in items[1].split(',')])

test = pd.DataFrame(np.array(test_list))
test.columns = ['id'] + ['s_'+str(i) for i in range(len(test_list[0])-1)]
test = reduce_mem_usage(test)
```

``` python
train.head()
```

| id   | s_0  | s_1      | s_2      | s_3      | s_4      | s_5      | s_6      | s_7      | s_8      | ...      | s_196 | s_197 | s_198 | s_199 | s_200 | s_201 | s_202 | s_203 | s_204 | label |      |
| :--- | :--- | :------- | :------- | :------- | :------- | :------- | :------- | :------- | :------- | :------- | :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- | ---- |
| 0    | 0.0  | 0.991211 | 0.943359 | 0.764648 | 0.618652 | 0.379639 | 0.190796 | 0.040222 | 0.026001 | 0.031708 | ...   | 0.0   | 0.0   | 0.0   | 0.0   | 0.0   | 0.0   | 0.0   | 0.0   | 0.0   | 0.0  |
| 1    | 1.0  | 0.971680 | 0.929199 | 0.572754 | 0.178467 | 0.122986 | 0.132324 | 0.094421 | 0.089600 | 0.030487 | ...   | 0.0   | 0.0   | 0.0   | 0.0   | 0.0   | 0.0   | 0.0   | 0.0   | 0.0   | 0.0  |
| 2    | 2.0  | 1.000000 | 0.958984 | 0.701172 | 0.231812 | 0.000000 | 0.080688 | 0.128418 | 0.187500 | 0.280762 | ...   | 0.0   | 0.0   | 0.0   | 0.0   | 0.0   | 0.0   | 0.0   | 0.0   | 0.0   | 2.0  |
| 3    | 3.0  | 0.975586 | 0.934082 | 0.659668 | 0.249878 | 0.237061 | 0.281494 | 0.249878 | 0.249878 | 0.241455 | ...   | 0.0   | 0.0   | 0.0   | 0.0   | 0.0   | 0.0   | 0.0   | 0.0   | 0.0   | 0.0  |
| 4    | 4.0  | 0.000000 | 0.055817 | 0.261230 | 0.359863 | 0.433105 | 0.453613 | 0.499023 | 0.542969 | 0.616699 | ...   | 0.0   | 0.0   | 0.0   | 0.0   | 0.0   | 0.0   | 0.0   | 0.0   | 0.0   | 2.0  |

``` python
test.head()
```

|      | id       | s_0      | s_1      | s_2      | s_3      | s_4      | s_5      | s_6      | s_7      | s_8      | ...  | s_195    | s_196    | s_197    | s_198    | s_199    | s_200    | s_201    | s_202    | s_203    | s_204   |
| :--- | :------- | :------- | :------- | :------- | :------- | :------- | :------- | :------- | :------- | :------- | :--- | :------- | :------- | :------- | :------- | :------- | :------- | :------- | :------- | :------- | :------ |
| 0    | 100000.0 | 0.991699 | 1.000000 | 0.631836 | 0.136230 | 0.041412 | 0.102722 | 0.120850 | 0.123413 | 0.107910 | ...  | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.00000 |
| 1    | 100001.0 | 0.607422 | 0.541504 | 0.340576 | 0.000000 | 0.090698 | 0.164917 | 0.195068 | 0.168823 | 0.198853 | ...  | 0.389893 | 0.386963 | 0.367188 | 0.364014 | 0.360596 | 0.357178 | 0.350586 | 0.350586 | 0.350586 | 0.36377 |
| 2    | 100002.0 | 0.975098 | 0.670898 | 0.686523 | 0.708496 | 0.718750 | 0.716797 | 0.720703 | 0.701660 | 0.596680 | ...  | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.00000 |
| 3    | 100003.0 | 0.995605 | 0.916992 | 0.520996 | 0.000000 | 0.221802 | 0.404053 | 0.490479 | 0.527344 | 0.518066 | ...  | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.00000 |
| 4    | 100004.0 | 1.000000 | 0.888184 | 0.745605 | 0.531738 | 0.380371 | 0.224609 | 0.091125 | 0.057648 | 0.003914 | ...  | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.00000 |

### 训练数据/测试数据准备

``` python
x_train = train.drop(['id','label'], axis=1)
y_train = train['label']
x_test=test.drop(['id'], axis=1)
```

### 模型训练

``` python
def abs_sum(y_pre,y_tru):
    y_pre=np.array(y_pre)
    y_tru=np.array(y_tru)
    loss=sum(sum(abs(y_pre-y_tru)))
    return loss
```

``` python
def cv_model(clf, train_x, train_y, test_x, clf_name):
    folds = 5
    seed = 2021
    kf = KFold(n_splits=folds, shuffle=True, random_state=seed)
    test = np.zeros((test_x.shape[0],4))

    cv_scores = []
    onehot_encoder = OneHotEncoder(sparse=False)
    for i, (train_index, valid_index) in enumerate(kf.split(train_x, train_y)):
        print('************************************ {} ************************************'.format(str(i+1)))
        trn_x, trn_y, val_x, val_y = train_x.iloc[train_index], train_y[train_index], train_x.iloc[valid_index], train_y[valid_index]
        
        if clf_name == "lgb":
            train_matrix = clf.Dataset(trn_x, label=trn_y)
            valid_matrix = clf.Dataset(val_x, label=val_y)

            params = {
                'boosting_type': 'gbdt',
                'objective': 'multiclass',
                'num_class': 4,
                'num_leaves': 2 ** 5,
                'feature_fraction': 0.8,
                'bagging_fraction': 0.8,
                'bagging_freq': 4,
                'learning_rate': 0.1,
                'seed': seed,
                'nthread': 28,
                'n_jobs':24,
                'verbose': -1,
            }

            model = clf.train(params, 
                      train_set=train_matrix, 
                      valid_sets=valid_matrix, 
                      num_boost_round=2000, 
                      verbose_eval=100, 
                      early_stopping_rounds=200)
            val_pred = model.predict(val_x, num_iteration=model.best_iteration)
            test_pred = model.predict(test_x, num_iteration=model.best_iteration) 
            
        val_y=np.array(val_y).reshape(-1, 1)
        val_y = onehot_encoder.fit_transform(val_y)
        print('预测的概率矩阵为：')
        print(test_pred)
        test += test_pred
        score=abs_sum(val_y, val_pred)
        cv_scores.append(score)
        print(cv_scores)
    print("%s_scotrainre_list:" % clf_name, cv_scores)
    print("%s_score_mean:" % clf_name, np.mean(cv_scores))
    print("%s_score_std:" % clf_name, np.std(cv_scores))
    test=test/kf.n_splits

    return test
```

``` python
def lgb_model(x_train, y_train, x_test):
    lgb_test = cv_model(lgb, x_train, y_train, x_test, "lgb")
    return lgb_test
```

``` python
lgb_test = lgb_model(x_train, y_train, x_test)
```

``` python
temp=pd.DataFrame(lgb_test)
result=pd.read_csv('sample_submit.csv')
result['label_0']=temp[0]
result['label_1']=temp[1]
result['label_2']=temp[2]
result['label_3']=temp[3]
result.to_csv('submit.csv',index=False)
```

lgb_scotrainre_list: [579.1476207255504, 604.2307776963925, 555.3013640683625, 605.9808495854536, 570.2883889772514]

lgb_score_mean: 582.9898002106021 

lgb_score_std: 19.608697878368307

Baseline 提交到天池的结果是 score:555.0982

