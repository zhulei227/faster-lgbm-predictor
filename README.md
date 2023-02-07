### 0.介绍

加速lgbm对单条数据的预测，提速X4倍左右


```python
import pandas as pd
import numpy as np
import lightgbm as lgb
from faster_lgbm_predictor_single import FasterLgbmSinglePredictor
from faster_lgbm_predictor_multiclass import FasterLgbmMulticlassPredictor
```

### 1.加载数据


```python
df=pd.read_csv("./data/train.csv")
```


```python
del df["PassengerId"]
del df["Name"]
del df["Sex"]
del df["Ticket"]
del df["Cabin"]
del df["Embarked"]
```


```python
df=df.fillna(0)
df.head(5)
```




<div>
 
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Fare</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>3</td>
      <td>22.0</td>
      <td>1</td>
      <td>0</td>
      <td>7.2500</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>1</td>
      <td>38.0</td>
      <td>1</td>
      <td>0</td>
      <td>71.2833</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>3</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>7.9250</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>1</td>
      <td>35.0</td>
      <td>1</td>
      <td>0</td>
      <td>53.1000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>3</td>
      <td>35.0</td>
      <td>0</td>
      <td>0</td>
      <td>8.0500</td>
    </tr>
  </tbody>
</table>
</div>




```python
target=df["Survived"].values
del df["Survived"]
```


```python
categorical_features=["Pclass","SibSp","Parch"]
```

### 2.1 二分类测试


```python
params={"objective":"binary","max_depth":2}
lgb_model=lgb.train(params=params,train_set=lgb.Dataset(data=df,label=target,categorical_feature=categorical_features),num_boost_round=16)
```

    [LightGBM] [Info] Number of positive: 342, number of negative: 549
    [LightGBM] [Warning] Auto-choosing row-wise multi-threading, the overhead of testing was 0.000900 seconds.
    You can set `force_row_wise=true` to remove the overhead.
    And if memory is not enough, you can set `force_col_wise=true`.
    [LightGBM] [Info] Total Bins 216
    [LightGBM] [Info] Number of data points in the train set: 891, number of used features: 5
    [LightGBM] [Info] [binary:BoostFromScore]: pavg=0.383838 -> initscore=-0.473288
    [LightGBM] [Info] Start training from score -0.473288
    

    D:\apps\Anaconda3\envs\autogluon\lib\site-packages\lightgbm\basic.py:2065: UserWarning: Using categorical_feature in Dataset.
      _log_warning('Using categorical_feature in Dataset.')
    


```python
faster_lgbm_predictor=FasterLgbmSinglePredictor(model=lgb_model.dump_model(),cache_num=10)
```


```python
ori_pred=lgb_model.predict(df)
```


```python
fast_pred=[]
for input_data in df.to_dict("records"):
    fast_pred.append(faster_lgbm_predictor.predict(input_data).get("score"))
fast_pred=np.asarray(fast_pred)
```


```python
np.sum(np.abs(fast_pred-ori_pred))
```




    0.0



### 2.2 多分类测试


```python
params={"objective":"multiclass","max_depth":2,"num_class":2}
lgb_model=lgb.train(params=params,train_set=lgb.Dataset(data=df,label=target,categorical_feature=categorical_features),num_boost_round=16)
```

    [LightGBM] [Warning] Auto-choosing row-wise multi-threading, the overhead of testing was 0.000781 seconds.
    You can set `force_row_wise=true` to remove the overhead.
    And if memory is not enough, you can set `force_col_wise=true`.
    [LightGBM] [Info] Total Bins 216
    [LightGBM] [Info] Number of data points in the train set: 891, number of used features: 5
    [LightGBM] [Info] Start training from score -0.484246
    [LightGBM] [Info] Start training from score -0.957534
    


```python
ori_pred=pd.DataFrame(lgb_model.predict(df))
```


```python
faster_lgbm_predictor=FasterLgbmMulticlassPredictor(model=lgb_model.dump_model(),cache_num=10)
```


```python
fast_pred=[]
for input_data in df.to_dict("records"):
    fast_pred.append(faster_lgbm_predictor.predict(input_data).get("score"))
fast_pred=pd.DataFrame(fast_pred)
```


```python
error_value=0
for col in fast_pred.columns:
    error_value+=np.sum(np.abs(fast_pred[col]-ori_pred[col]))
error_value
```




    8.013034680232067e-14



### 2.3 回归测试


```python
params={"objective":"regression","max_depth":2}
lgb_model=lgb.train(params=params,train_set=lgb.Dataset(data=df,label=target,categorical_feature=categorical_features),num_boost_round=16)
```

    [LightGBM] [Warning] Auto-choosing row-wise multi-threading, the overhead of testing was 0.015627 seconds.
    You can set `force_row_wise=true` to remove the overhead.
    And if memory is not enough, you can set `force_col_wise=true`.
    [LightGBM] [Info] Total Bins 216
    [LightGBM] [Info] Number of data points in the train set: 891, number of used features: 5
    [LightGBM] [Info] Start training from score 0.383838
    


```python
faster_lgbm_predictor=FasterLgbmSinglePredictor(model=lgb_model.dump_model(),cache_num=10)
```


```python
ori_pred=lgb_model.predict(df)
```


```python
fast_pred=[]
for input_data in df.to_dict("records"):
    fast_pred.append(faster_lgbm_predictor.predict(input_data).get("score"))
fast_pred=np.asarray(fast_pred)
```


```python
np.sum(np.abs(fast_pred-ori_pred))
```




    0.0



### 2.4 指数分布回归测试¶


```python
params={"objective":"tweedie","max_depth":2}
lgb_model=lgb.train(params=params,train_set=lgb.Dataset(df,target,categorical_feature=categorical_features),num_boost_round=16)
```

    [LightGBM] [Warning] Auto-choosing col-wise multi-threading, the overhead of testing was 0.012412 seconds.
    You can set `force_col_wise=true` to remove the overhead.
    [LightGBM] [Info] Total Bins 216
    [LightGBM] [Info] Number of data points in the train set: 891, number of used features: 5
    [LightGBM] [Info] Start training from score -0.957534
    


```python
faster_lgbm_predictor=FasterLgbmSinglePredictor(model=lgb_model.dump_model(),cache_num=10)
```


```python
ori_pred=lgb_model.predict(df)
```


```python
fast_pred=[]
for input_data in df.to_dict("records"):
    fast_pred.append(faster_lgbm_predictor.predict(input_data).get("score"))
fast_pred=np.asarray(fast_pred)
```


```python
np.sum(np.abs(fast_pred-ori_pred))
```




    0.0



### 性能对比


```python
from tqdm import tqdm
```


```python
new_data=df.to_dict("records")
for data in tqdm(new_data):
    faster_lgbm_predictor.predict(data)
```

    100%|███████████████████████████████████████████████████████████████████████████████| 891/891 [00:02<00:00, 425.57it/s]
    


```python
new_data=[]
for i in list(range(len(df))):
    new_data.append(df[i:i+1])
for data in tqdm(new_data):
    lgb_model.predict(data)
```

    100%|████████████████████████████████████████████████████████████████████████████████| 891/891 [00:10<00:00, 88.66it/s]
    


```python

```
