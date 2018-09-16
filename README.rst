MLFeatureSelection
==================

|License: MIT| |PyPI version|

General features selection based on certain machine learning algorithm
and evaluation methods

**Divesity, Flexible and Easy to use**

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

    sf = sequence_selection.Select(Sequence = True, Random = True, Cross = False) 
    sf.ImportDF(df,label = 'Label') #import dataframe and label
    sf.ImportLossFunction(lossfunction, direction = 'ascend') #import loss function handle and optimize direction, 'ascend' for AUC, ACC, 'descend' for logloss etc.
    sf.InitialNonTrainableFeatures(notusable) #those features that is not trainable in the dataframe, user_id, string, etc
    sf.InitialFeatures(initialfeatures) #initial initialfeatures as list
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

-  format of validate and lossfunction

define your own:

**validate**: validation method in function , ie k-fold, last time
section valdate, random sampling validation, etc

**lossfunction**: model performance evaluation method, ie logloss, auc,
accuracy, etc

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

-  Demo contain all modulus can be found here
   (`demo <https://github.com/duxuhao/Feature-Selection/blob/master/Demo.py>`__)

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

Function Parameters
-------------------

sf = sequence\_selection.Select(Sequence=True, Random=True, Cross=True)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Parameters:
^^^^^^^^^^^

**Sequence** (*bool*, optional, (defualt=True)) - switch for sequence
selection selection include forward,backward and simulate anneal
selection

**Random** (*bool, optional, (defualt=True)*) - switch for randomly
selection of features combination

**Cross** (*bool*, optional, (defualt=True)) - switch for cross term
generate, need to set sf.ImportCrossMethod() after

sf.ImportDF(df,label)
~~~~~~~~~~~~~~~~~~~~~

Parameters:
^^^^^^^^^^^

**df** (*pandas.DataFrame*) - dataframe includes include all features

**label** (*str*) - name of the label column

sf.ImportLossFunction(lossfunction,direction)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Parameters:
^^^^^^^^^^^

**lossfunction** (*function handle*) - handle of the loss function,
function should return score as float (logloss, AUC, etc)

**direction** (*str,'ascend'/'descend'*) - direction to improve,
'descend' for logloss, 'ascend' for AUC, etc

sf.InitialFeatures(features)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Parameters:
^^^^^^^^^^^

**features** (*list, optional, (defualt=[])*) - list of initial features
combination, empty list will drive code to start from nothing list with
all trainable features will drive code to start backward searching at
the beginning

sf.InitialNonTrainableFeatures(features) #only for sequence selection
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Parameters:
^^^^^^^^^^^

**features** (*list*) - list of features that not trainable (labelname,
string, datetime, etc)

sf.GenerateCol(key=None,selectstep=1) #only for sequence selection
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Parameters:
^^^^^^^^^^^

**key** (*str, optional, default=None*) - only the features with keyword
will be seleted, default to be None

**selectstep** (*int, optional, default=1*) - value for features
selection step

sf.SelectRemoveMode(frac=1,batch=1,key='')
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Parameters:
^^^^^^^^^^^

**frac** (*float, optional, default=1*) - percentage of delete features
from all features default to be set as using the batch

**batch** (*int, optional, default=1*) - delete features quantity every
iteration

**key** (*str, optional, default=None*) - only delete the features with
keyword

sf.ImportCrossMethod(CrossMethod)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Parameters:
^^^^^^^^^^^

**CrossMethod** (*dict*) - different cross method like add, divide,
multiple and substraction

sf.AddPotentialFeatures(features)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Parameters:
^^^^^^^^^^^

**features** (*list*, optional, default=[]\_) - list of strong features,
switch for simulate anneal

sf.SetTimeLimit(TimeLimit=inf)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Parameters:
^^^^^^^^^^^

**TimeLimit** (*float, optional, default=inf*) - maximum running time,
unit in minute

sf.SetFeaturesLimit(FeaturesLimit=inf)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Parameters:
^^^^^^^^^^^

**FeaturesLimit** (int, optional, default=inf\_) - maximum feature
quantity

sf.SetClassifier(clf)
~~~~~~~~~~~~~~~~~~~~~

Parameters:
^^^^^^^^^^^

**clf** (*predictor*) - classfier or estimator, sklearn, xgboost,
lightgbm, etc. Need to match the validate function

sf.SetLogFile(logfile)
~~~~~~~~~~~~~~~~~~~~~~

Parameters:
^^^^^^^^^^^

**logfile** (*str*) - log file name

sf.run(validate)
~~~~~~~~~~~~~~~~

Parameters:
^^^^^^^^^^^

**validate** (*function handle*) - function return evaluation score and
predictor input features dataset X, label series Y, used features,
predictor, lossfunction handle

Algorithm details (selecting features based on greedy algorithm)
----------------------------------------------------------------

.. figure:: https://github.com/duxuhao/Feature-Selection/blob/master/Procedure0.png
   :alt: Procedure

   Procedure

.. |License: MIT| image:: https://img.shields.io/badge/License-MIT-yellow.svg
   :target: https://opensource.org/licenses/MIT
.. |PyPI version| image:: https://badge.fury.io/py/MLFeatureSelection.svg
   :target: https://pypi.org/project/MLFeatureSelection/
