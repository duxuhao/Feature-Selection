import LRS_SA_RGSS as LSR
import pandas as pd
import lightgbm as lgbm
from sklearn.metrics import log_loss

def modelscore(y_test, y_pred):
    """for setting up the evaluation score
    """
    return log_loss(y_test, y_pred)

""" The cross methods """
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

def prepareData():
    """prepare you dataset here"""
    df = pd.read_csv('data/train/trainb.csv')
    df = df[~pd.isnull(df.is_trade)]
    item_category_list_unique = list(np.unique(df.item_category_list))
    df.item_category_list.replace(item_category_list_unique, list(np.arange(len(item_category_list_unique))), inplace=True)
    return df
    
def main(temp, clf, CrossMethod, RecordFolder, test = False):
    # set up the data set first
    df = prepareData()
    # get the features for selection
    uselessfeatures = ['used','instance_id', 'item_property_list', 'context_id', 'context_timestamp', 'predict_category_property', 'is_trade'] # features not trainable
    ColumnName = obtaincol(df, uselessfeatures) #obtain columns withouth the useless features
    SearchColumns = ColumnName[:] # the search features library. if columnname == [] teh code will run the backward searching at the very beginning
    # start selecting
    a = LSR.LRS_SA_RGSS_combination(df = df,
                                    clf = clf,
                                    RecordFolder = RecordFolder,
                                    LossFunction = modelscore,
                                    label = 'is_trade',
                                    columnname = SearchColumns[:],  
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
    # algorithm group, add any sklearn type algorithm inside as a based selection algorithm
    # change the validation function in file LRS_SA_RGSS.py for your own case
    model = {'lgb6': lgbm.LGBMClassifier(random_state=1, num_leaves = 6, n_estimators=5000, max_depth=3, learning_rate = 0.05, n_jobs=8)}

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
                  'shop_score_service', 'shop_score_delivery', 'hour', 'day'] # start features combination
    main(temp,model[modelselect], CrossMethod, RecordFolder,test=False)
