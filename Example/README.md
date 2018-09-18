# MLFeatureSelection
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyPI version](https://badge.fury.io/py/MLFeatureSelection.svg)](https://pypi.org/project/MLFeatureSelection/)

General features selection based on certain machine learning algorithm and evaluation methods

## Modulus Usage

- sequence_selection

```python
from MLFeatureSelection import sequence_selection
from sklearn.linear_model import LogisticRegression

sf = sequence_selection.Select(Sequence = True, Random = True, Cross = False) 
sf.ImportDF(df,label = 'Label') #import dataframe and label
sf.ImportLossFunction(lossfunction, direction = 'ascend') #import loss function handle and optimize direction, 'ascend' for AUC, ACC, 'descend' for logloss etc.
sf.InitialNonTrainableFeatures(notusable) #those features that is not trainable in the dataframe, user_id, string, etc
sf.InitialFeatures(initialfeatures) #initial initialfeatures as list
sf.GenerateCol() #generate features for selection
sf.clf = LogisticRegression() #set the selected algorithm, can be any algorithm
sf.SetLogFile('record.log') #log file
sf.run(validate) #run with validation function, validate is the function handle of the validation function, return best features combination
```

- importance_selection

```python
from MLFeatureSelection import importance_selection
import xgboost as xgb

sf = importance_selection.Select() 
sf.ImportDF(df,label = 'Label') #import dataframe and label
sf.ImportLossFunction(lossfunction, direction = 'ascend') #import loss function and optimize direction
sf.InitialFeatures() #initial features, input
sf.SelectRemoveMode(batch = 2)
sf.clf = xgb.XGBClassifier() 
sf.SetLogFile('record.log') #log file
sf.run(validate) #run with validation function, return best features combination
```

- coherence_selection

```python
from MLFeatureSelection import coherence_selection
import xgboost as xgb

sf = coherence_selection.Select() 
sf.ImportDF(df,label = 'Label') #import dataframe and label
sf.ImportLossFunction(lossfunction, direction = 'ascend') #import loss function and optimize direction
sf.InitialFeatures() #initial features, input
sf.SelectRemoveMode(batch = 2)
sf.clf = xgb.XGBClassifier() 
sf.SetLogFile('record.log') #log file
sf.run(validate) #run with validation function, return best features combination
```

- log reader

```python
from MLFeatureSelection.tools import readlog

logfile = 'record.log'
logscore = 0.5 #any score in the logfile
features_combination = readlog(logfile, logscore)
```

- format of validate and lossfunction

define your own:

**validate**: validation method in function , ie k-fold, last time section valdate, random sampling validation, etc

**lossfunction**: model performance evaluation method, ie logloss, auc, accuracy, etc

```python
def validate(X, y, features, clf, lossfunction):
    """define your own validation function with 5 parameters
    input as X, y, features, clf, lossfunction
    clf is set by SetClassifier()
    lossfunction is import earlier
    features will be generate automatically
    function return score and trained classfier
    """
    clf.fit(X[features],y)
    y_pred = clf.predict(X[features])
    score = lossfuntion(y_pred,y)
    return score, clf
    
def lossfunction(y_pred, y_test):
    """define your own loss function with y_pred and y_test
    return score
    """
    return np.mean(y_pred == y_test)
```

    
## DEMO

More examples are added in example folder include:

- Demo contain all modulus can be found here ([demo](https://github.com/duxuhao/Feature-Selection/blob/master/Demo.py))

- Simple Titanic with 5-fold validation and evaluated by accuracy ([demo](https://github.com/duxuhao/Feature-Selection/tree/master/example/titanic))

- Demo for S1, S2 score improvement in JData 2018 predict purchase time competition ([demo](https://github.com/duxuhao/Feature-Selection/tree/master/example/JData2018))

- Demo for IJCAI 2018 CTR prediction ([demo](https://github.com/duxuhao/Feature-Selection/tree/master/example/IJCAI-2018))
