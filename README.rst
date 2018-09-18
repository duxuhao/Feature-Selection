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

Modulus in version 0.0.8.2
--------------------------

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

`Example <https://github.com/duxuhao/Feature-Selection/blob/master/Example>`__

This features selection method achieved
---------------------------------------

-  **1st** in Rong360

-- https://github.com/duxuhao/rong360-season2

-  **6th** in JData-2018

-- https://github.com/duxuhao/JData-2018

-  **12nd** in IJCAI-2018 1st round

-- https://github.com/duxuhao/IJCAI-2018-2

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

Function Parameters
-------------------

`Parameters <https://github.com/duxuhao/Feature-Selection/blob/master/MLFeatureSelection>`__

Algorithm details
-----------------

`Details <https://github.com/duxuhao/Feature-Selection/blob/master/Algorithms_Graphs>`__

.. |License: MIT| image:: https://img.shields.io/badge/License-MIT-yellow.svg
   :target: https://opensource.org/licenses/MIT
.. |PyPI version| image:: https://badge.fury.io/py/MLFeatureSelection.svg
   :target: https://pypi.org/project/MLFeatureSelection/
