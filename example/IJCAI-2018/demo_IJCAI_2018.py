from MLFeatureSelection import sequence_selection as ss
from sklearn.metrics import log_loss
import lightgbm as lgbm
import pandas as pd
import numpy as np

def prepareData():
    """prepare you dataset here"""
    df = pd.read_csv('data/train/trainb.csv')
    df = df[~pd.isnull(df.is_trade)]
    item_category_list_unique = list(np.unique(df.item_category_list))
    df.item_category_list.replace(item_category_list_unique, list(np.arange(len(item_category_list_unique))), inplace=True)
    return df

def modelscore(y_test, y_pred):
    """set up the evaluation score"""
    return log_loss(y_test, y_pred)

def validation(X,y,features, clf,lossfunction):
    """set up your validation method"""
    totaltest = 0
    for D in [24]:
        T = (X.day != D)
        X_train, X_test = X[T], X[~T]
        X_train, X_test = X_train[features], X_test[features]
        y_train, y_test = y[T], y[~T]
        clf.fit(X_train,y_train, eval_set = [(X_train, y_train), (X_test, y_test)], eval_metric='logloss', verbose=False,early_stopping_rounds=200)
        totaltest += lossfunction(y_test, clf.predict_proba(X_test)[:,1])
    totaltest /= 1.0
    return totaltest,clf

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

def main():
    sf = ss.Select(Sequence = True, Random = True, Cross = False) #select the way you want to process searching
    sf.ImportDF(prepareData(),label = 'is_trade')
    sf.ImportLossFunction(modelscore,direction = 'descend')
    sf.ImportCrossMethod(CrossMethod)
    sf.InitialNonTrainableFeatures(['used','instance_id', 'item_property_list', 'context_id', 'context_timestamp', 'predict_category_property', 'is_trade'])
    sf.InitialFeatures(['item_category_list', 'item_price_level','item_sales_level','item_collected_level', 'item_pv_level','day'])
    sf.GenerateCol(key = 'mean', selectstep = 2)
    sf.SetSample(0.1, samplemode = 0, samplestate = 0)
#    sf.SetFeaturesLimit(5)
    sf.SetTimeLimit(1)
    sf.clf = lgbm.LGBMClassifier(random_state=1, num_leaves = 6, n_estimators=5000, max_depth=3, learning_rate = 0.05, n_jobs=8)
    sf.SetLogFile('recordml.log')
    sf.run(validation)

if __name__ == "__main__":
    main()
