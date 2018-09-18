# MLFeatureSelection
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyPI version](https://badge.fury.io/py/MLFeatureSelection.svg)](https://pypi.org/project/MLFeatureSelection/)

General features selection based on certain machine learning algorithm and evaluation methods

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
