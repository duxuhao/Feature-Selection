from sklearn.linear_model import LogisticRegression
from MLFeatureSelection import FeatureSelection as FS
from sklearn.metrics import log_loss
import lightgbm as lgbm
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold

def prepareData():
    df = pd.read_csv('clean_train.csv')
    Title = list(np.unique(df.Title))
    df.Title.replace(Title, list(np.arange(len(Title))), inplace=True)
    return df

def modelscore(y_test, y_pred):
    return np.mean(y_test == y_pred)

def validation(X,y, features, clf, lossfunction):
    totaltest = []
    kf = KFold(5)
    for train_index, test_index in kf.split(X):
        X_train, X_test = X.ix[train_index,:][features], X.ix[test_index,:][features]
        y_train, y_test = y[train_index], y[test_index]
        #clf.fit(X_train, y_train, eval_set = [(X_train, y_train), (X_test, y_test)], eval_metric='logloss', verbose=False,early_stopping_rounds=50)
        clf.fit(X_train, y_train)
        totaltest.append(lossfunction(y_test, clf.predict(X_test)))
    return np.mean(totaltest)

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
    sf = FS.Select(Sequence = True, Random = True, Cross = True)
    sf.ImportDF(prepareData(),label = 'Survived')
    sf.ImportLossFunction(modelscore,direction = 'ascend')
    sf.ImportCrossMethod(CrossMethod)
    sf.NonTrainableFeatures = ['Survived']
    sf.InitialFeatures([])
    sf.PotentialAdd = ['Pclass']
    #sf.clf = lgbm.LGBMClassifier(random_state=1, num_leaves = 6, n_estimators=5000, max_depth=3, learning_rate = 0.05, n_jobs=1)
    sf.clf = LogisticRegression()
    sf.logfile = 'record.log'
    sf.run(validation)

if __name__ == "__main__":
    main()
