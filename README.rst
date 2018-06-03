Features Selection
==================

General features selection based on certain machine learning algorithm
and evaluation methods

Divesity, Flexible and Easy to use

More features selection method will be included in the future!

Quick Installation
------------------

.. code:: python

    pip3 install MLFeatureSelection

Modulus in version 0.0.7
------------------------

-  Modulus for selecting features based on greedy algorithm (from
   MLFeatureSelection import sequence\_selection)

-  Modulus for removing features based on features importance (from
   MLFeatureSelection import importance\_selection)

-  Modulus for removing features based on correlation coefficient (from
   MLFeatureSelection import coherence\_selection)

-  Modulus for reading the features combination from log file (from
   MLFeatureSelection.tools import readlog)

Modulus Usage
-------------

-  sequence\_selection

.. code:: python

    from MLFeatureSelection import sequence_selection
    from sklearn.linear_model import LogisticRegression

    sf = sequence_selection.Select(Sequence = True, Random = True, Cross = True) 
    sf.ImportDF(df,label = 'Label') #import dataframe and label
    sf.ImportLossFunction(lossfunction, direction = 'ascend') #import loss function handle and optimize direction, 'ascend' for AUC, ACC, 'descend' for logloss etc.
    sf.InitialNonTrainableFeatures(notusable) #those features that is not trainable in the dataframe, user_id, string, etc
    sf.InitialFeatures(initialfeatures) #initial initialfeatures as list
    sf.SelectRemoveMode(batch = 2)
    sf.GenerateCol() #generate features for selection
    sf.clf = LogisticRegression() #set the selected algorithm, can be any algorithm
    sf.SetLogFile('record.log') #log file
    sf.run(validate) #run with validation function, validate is the function handle of the validation function, return best features combination

-  importance\_selection

.. code:: python

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

-  coherence\_selection

.. code:: python

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

-  log reader

.. code:: python

    from MLFeatureSelection.tools import readlog

    logfile = 'record.log'
    logscore = 0.5 #any score in the logfile
    features_combination = readlog(logfile, logscore)

Function Parameters
-------------------

**sf.ImportDF(df,label)**

::

    df: pd.DataFrame, include all features    
    label: str, name of the label column

**sf.ImportLossFunction(lossfunction,direction)**

::

    lossfunction: handle of the loss function, function return score as scalar value (logloss, AUC, etc)    
    direction: 'ascend'/'descend', direction to improve

**sf.InitialFeatures(features)**

::

    features: list of initial features combination,     
              empty list will drive code to start from nothing    
              list with all trainable features will drive code               
              to start backward searching at the beginning
              

**sf.InitialNonTrainableFeatures(features)** #only for sequence
selection

::

    features: list of features that not trainable (string, datetime, etc)

**sf.GenerateCol(key=None,selectstep=1)** #only for sequence selection

::

    key: str for the selected features, only the features with keyword will be seleted,         
         default to be None         
    selectstep: int, value for features selection step, default to be 1

**sf.SelectRemoveMode(frac=1,batch=1,key='')**

::

    frac: float, percentage of delete features from all features    
          default to be 1 as using the batch          
    batch: int, delete features quantity every iteration    
    key: str, only delete the features with keyword

**sf.SetTimeLimit(TimeLimit)**

::

    TimeLimit: float, maximum running time, unit in minute

**sf.SetFeaturesLimit(FeaturesLimit)**

::

    FeaturesLimit: int, maximum feature quantity

**sf.SetClassifier(clf)**

::

    clf: classfier or estimator, sklearn, xgboost, lightgbm, etc

**sf.SetLogFile(logfile)**

::

    logfile: str, log file name

**sf.run(validate)**

::

    validate: function handle with score and classifier return

.. code:: python

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

DEMO
----

More examples are added in example folder include:

-  Simple Titanic with 5-fold validation and evaluated by accuracy
   (`demo <https://github.com/duxuhao/Feature-Selection/tree/master/example/titanic>`__)

-  Demo for S1, S2 score improvement in JData 2018 predict purchase time
   competition
   (`demo <https://github.com/duxuhao/Feature-Selection/tree/master/example/JData2018>`__)

-  Demo for IJCAI 2018 CTR prediction
   (`demo <https://github.com/duxuhao/Feature-Selection/tree/master/example/IJCAI-2018>`__)

PLAN
----

-  better API introduction will be completed next before the end of
   06/2018

This features selection method achieved
---------------------------------------

-  **1st** in Rong360

-- https://github.com/duxuhao/rong360-season2

-  **Temporary Top 10** in JData-2018 (Peter Du)

-  **12nd** in IJCAI-2018 1st round

Algorithm details (selecting features based on greedy algorithm)
----------------------------------------------------------------

.. figure:: https://github.com/duxuhao/Feature-Selection/blob/master/Procedure0.png
   :alt: Procedure

   Procedure
