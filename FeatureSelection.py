import LRS_SA_RGSS as LSR
from preprocessing import preprocessing
from utility import *
import pandas as pd
import xgboost as xgb
import lightgbm as lgbm
from sklearn.metrics import log_loss

def modelscore(y_test, y_pred):
    """for setting up the evaluation score
    """
    return log_loss(y_test, y_pred)

def add(x,y):
    return x + y

def substract(x,y):
    return x - y

def times(x,y):
    return x * y

def divide(x,y):
    return (x + 0.001)/(y + 0.001)

def obtaincol(df, delete):
    """ for getting rid of the useless columns in the dataset
    """
    ColumnName = list(df.columns)
    for i in delete:
        if i in ColumnName:
            ColumnName.remove(i)
    return ColumnName

def main(temp, clf, CrossMethod, RecordFolder, test = False):
    # set up the data set first
    df = pd.read_csv('data/train/trainb.csv')
    df = df[~pd.isnull(df.is_trade)]
    item_category_list_unique = list(np.unique(df.item_category_list))
    df.item_category_list.replace(item_category_list_unique, list(np.arange(len(item_category_list_unique))), inplace=True)
    # get the features for selection
    uselessfeatures = ['used','instance_id', 'item_property_list', 'context_id', 'context_timestamp', 'predict_category_property', 'is_trade']
    ColumnName = obtaincol(df, uselessfeatures) # + addcol #obtain columns withouth the useless features
    print(ColumnName)
    # start selecting
    a = LSR.LRS_SA_RGSS_combination(df = df,
                                    clf = clf,
                                    RecordFolder = RecordFolder,
                                    LossFunction = modelscore,
                                    label = 'is_trade',
                                    columnname = ColumnName[1::2], # the pattern for selection
                                    start = temp,
                                    CrossMethod = CrossMethod, # your cross term method
                                    PotentialAdd = [] # potential feature for Simulated Annealing
                                    )
    try:
        a.run()
    finally:
        with open(RecordFolder, 'a') as f:
            f.write('\n{}\n%{}%\n'.format(type,'-'*60))

if __name__ == "__main__":
    model = {'xgb': xgb.XGBClassifier(seed = 1, max_depth = 5, n_estimators = 2000, nthread = -1),
             'lgb': lgbm.LGBMClassifier(random_state=1,num_leaves = 29, n_estimators=1000),
             'lgb2': lgbm.LGBMClassifier(random_state=1,num_leaves = 29, max_depth=5, n_estimators=1000),
             'lgb3': lgbm.LGBMClassifier(random_state=1, num_leaves = 6, n_estimators=1000,max_depth=3,learning_rate = 0.09, n_jobs=30),
             'lgb4': lgbm.LGBMClassifier(random_state=1, num_leaves = 6, n_estimators=5000,max_depth=3,learning_rate = 0.095, n_jobs=30),
             'lgb5': lgbm.LGBMClassifier(random_state=1, num_leaves = 13, n_estimators=5000,max_depth=4,learning_rate = 0.05, n_jobs=30),
             'lgb6': lgbm.LGBMClassifier(random_state=1, num_leaves = 6, n_estimators=5000,max_depth=3,learning_rate = 0.05, n_jobs=8)
            } # algorithm group

    CrossMethod = {'+':add,
                   '-':substract,
                   '*':times,
                   '/':divide,}

    RecordFolder = 'record.log' # result record file
    modelselect = 'lgb6' # selected algorithm

    temp = ['item_category_list', 'item_price_level',
                  'item_sales_level',
                  'item_collected_level', 'item_pv_level',
                  'user_gender_id', 'user_age_level', 'user_occupation_id', 'user_star_level',
                  'context_page_id', 'shop_review_num_level', 'shop_review_positive_rate',
                  'shop_score_service', 'shop_score_delivery', 'hour', 'day', 'user_id_query_day_hour',
                  'shop_id'] # start features combination
    main(temp,model[modelselect], CrossMethod, RecordFolder,test=False)
