# MLFeatureSelection
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyPI version](https://badge.fury.io/py/MLFeatureSelection.svg)](https://pypi.org/project/MLFeatureSelection/)

General features selection based on certain machine learning algorithm and evaluation methods

**Divesity, Flexible and Easy to use**

More features selection method will be included in the future!

## Quick Installation

```python
pip3 install MLFeatureSelection
```

## Modulus in version 0.0.7

- Modulus for selecting features based on greedy algorithm (from MLFeatureSelection import sequence_selection)

- Modulus for removing features based on features importance (from MLFeatureSelection import importance_selection)

- Modulus for removing features based on correlation coefficient (from MLFeatureSelection import coherence_selection)

- Modulus for reading the features combination from log file (from MLFeatureSelection.tools import readlog)

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

## PLAN

- better API introduction will be completed next before the end of 06/2018


## This features selection method achieved

- **1st** in Rong360

   -- https://github.com/duxuhao/rong360-season2
   
- **Temporary Top 10** in JData-2018 (Peter Du)

- **12nd** in IJCAI-2018 1st round

## Function Parameters

### sf = sequence_selection.Select(Sequence=True, Random=True, Cross=True)

#### Parameters:  

**Sequence** (_bool_, optional, (defualt=True)) - switch for sequence selection selection include forward,backward and simulate anneal selection
                            
**Random** (_bool, optional, (defualt=True)_) - switch for randomly selection of features combination
                          
**Cross** (_bool_, optional, (defualt=True)) - switch for cross term generate, need to set sf.ImportCrossMethod() after

### sf.ImportDF(df,label)
    
#### Parameters:  

**df** (_pandas.DataFrame_) - dataframe includes include all features  
               
**label** (_str_) - name of the label column
    
### sf.ImportLossFunction(lossfunction,direction)

#### Parameters:  

**lossfunction** (_function handle_) - handle of the loss function, function  should return score as float (logloss, AUC, etc)    
                                
**direction** (_str,'ascend'/'descend'_) - direction to improve, 'descend' for logloss, 'ascend' for AUC, etc
    
### sf.InitialFeatures(features)

#### Parameters:  

**features** (_list, optional, (defualt=[])_) - list of initial features combination, empty list will drive code to start from nothing list with all trainable features will drive code to start backward searching at the beginning
              
### sf.InitialNonTrainableFeatures(features) #only for sequence selection

#### Parameters:  

**features** (_list_) - list of features that not trainable (labelname, string, datetime, etc)

### sf.GenerateCol(key=None,selectstep=1) #only for sequence selection

#### Parameters:  

**key** (_str, optional, default=None_) - only the features with keyword will be seleted, default to be None         
                       
**selectstep** (_int, optional, default=1_) - value for features selection step
    
### sf.SelectRemoveMode(frac=1,batch=1,key='')

#### Parameters:  

**frac** (_float, optional, default=1_) - percentage of delete features from all features default to be set as using the batch   
                        
**batch** (_int, optional, default=1_) - delete features quantity every iteration  
               
**key** (_str, optional, default=None_) - only delete the features with keyword
    
### sf.ImportCrossMethod(CrossMethod)

#### Parameters:  

**CrossMethod** (_dict_) - different cross method like add, divide, multiple and substraction
    
### sf.AddPotentialFeatures(features)

#### Parameters:  

**features** (_list_, optional, default=[]_) - list of strong features, switch for simulate anneal
    
### sf.SetTimeLimit(TimeLimit=inf)

#### Parameters:  

**TimeLimit** (_float, optional, default=inf_) - maximum running time, unit in minute
    
### sf.SetFeaturesLimit(FeaturesLimit=inf)

#### Parameters:  

**FeaturesLimit** (int, optional, default=inf_) - maximum feature quantity
    
### sf.SetClassifier(clf)

#### Parameters:  

**clf** (_predictor_) -  classfier or estimator, sklearn, xgboost, lightgbm, etc. Need to match the validate function

### sf.SetLogFile(logfile)

#### Parameters:  

**logfile** (_str_) - log file name
    
### sf.run(validate)

#### Parameters:  

**validate** (_function handle_) - function return evaluation score and predictor input features dataset X, label series Y, used features, predictor, lossfunction handle

## Algorithm details (selecting features based on greedy algorithm)

![Procedure](https://github.com/duxuhao/Feature-Selection/blob/master/Procedure0.png)
