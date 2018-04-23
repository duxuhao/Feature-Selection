# Features Selection
This code is based on the IJCAI-2018 but can tune easily for other dataset

## How to run
- modify the read dataset in FeatureSelection.py

- modify the features combination you want to start with in temp variable in FeatureSelection.py

- modify the useless features in FeatureSelection.py

- add the potential features you want to add in

- select your algorithm and recorded file name

- change the validation in function k_fold in file LRS_SA_RGSS.py

- change the evaluation operator in function ScoreUpdate() in LRS_SA_RGSS.py (> or <)

- run the FeatureSelection.py

- check the record file to see the result

- This code take a while to run, you can stop it any time and restart by replace the best features combination in temp

## This features selection method achieved

- **1st** in Rong360

   -- https://github.com/duxuhao/rong360-season2

- **12nd** in IJCAI-2018 1st round

## Algorithm details

![Procedure](https://github.com/duxuhao/Feature-Selection/blob/master/Procedure.png)
