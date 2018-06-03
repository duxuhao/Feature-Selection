This demo is based on the IJCAI-2018 data moning competitions

- Import library from FeatureSelection.py and also other necessary library

```python
from MLFeatureSelection import sequence_selection as FS 
from sklearn.metrics import log_loss
import lightgbm as lgbm
import pandas as pd
import numpy as np
```

- Generate for dataset

```python
def prepareData():
    df = pd.read_csv('data/train/trainb.csv')
    df = df[~pd.isnull(df.is_trade)]
    item_category_list_unique = list(np.unique(df.item_category_list))
    df.item_category_list.replace(item_category_list_unique, list(np.arange(len(item_category_list_unique))), inplace=True)
    return df
```

- Define your loss function

```python
def modelscore(y_test, y_pred):
    return log_loss(y_test, y_pred)
```

- Define the way to validate

```python
def validation(X,y, features, clf,lossfunction):
    totaltest = 0
    for D in [24]:
        T = (X.day != D)
        X_train, X_test = X[T], X[~T]
        X_train, X_test = X_train[features], X_test[features]
        y_train, y_test = y[T], y[~T]
        clf.fit(X_train,y_train, eval_set = [(X_train, y_train), (X_test, y_test)], eval_metric='logloss', verbose=False,early_stopping_rounds=200) #the train method must match your selected algorithm
        totaltest += lossfunction(y_test, clf.predict_proba(X_test)[:,1])
    totaltest /= 1.0
    return totaltest
```

- Define the cross method (required when *Cross = True*)

```python
def add(x,y):
    return x + y

def substract(x,y):
    return x - y

def times(x,y):
    return x * y

def divide(x,y):
    return (x + 0.001)/(y + 0.001)

CrossMethod = {'+':add,
               '-':substract,
               '*':times,
               '/':divide,}
```

- Initial the seacher with customized procedure (sequence + random + cross)

```python
sf = FS.Select(Sequence = False, Random = True, Cross = False) #select the way you want to process searching
```

- Import loss function

```python
sf.ImportLossFunction(modelscore, direction = 'descend')
```

- Import dataset

```python
sf.ImportDF(prepareData(), label = 'is_trade')
```

- Import cross method (required when *Cross = True*)

```python
sf.ImportCrossMethod(CrossMethod)
```

- Define non-trainable features

```python
sf.InitialNonTrainableFeatures(['used','instance_id', 'item_property_list', 'context_id', 'context_timestamp', 'predict_category_property', 'is_trade'])
```

- Define initial features' combination

```python
sf.InitialFeatures(['item_category_list', 'item_price_level','item_sales_level','item_collected_level', 'item_pv_level'])
```

- Generate feature library, can specific certain key word and selection step

```python

sf.GenerateCol(key = 'mean', selectstep = 2) #can iterate different features set
```

- Set maximum features quantity

```python
sf.SetFeaturesLimit(40) #maximum number of features
```

- Set maximum time limit (in minutes)

```python
sf.SetTimeLimit(100) #maximum running time in minutes
```

- Set sample ratio of total dataset, when samplemode equals to 0, running the same subset, when samplemode equals to 1, subset will be different each time

```python
sf.SetSample(0.1, samplemode = 0)
```

- Define algorithm

```python
sf.clf = lgbm.LGBMClassifier(random_state=1, num_leaves = 6, n_estimators=5000, max_depth=3, learning_rate = 0.05, n_jobs=8)
```

- Define log file name

```python
sf.SetLogFile('record.log')
```

- Run with self-define validate method

```python
sf.run(validation)
```

see complete code in demo_IJCAI_2018.py
