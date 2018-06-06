from sklearn.linear_model import LogisticRegression
from MLFeatureSelection import sequence_selection as ss
from sklearn.metrics import log_loss
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
    return np.mean(totaltest), clf

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
    sf = ss.Select(Sequence = True, Random = False, Cross = True)
    sf.ImportDF(prepareData(),label = 'Survived')
    sf.ImportLossFunction(modelscore,direction = 'ascend')
    sf.ImportCrossMethod(CrossMethod)
    sf.InitialNonTrainableFeatures(['Survived'])
    sf.InitialFeatures([])
    sf.GenerateCol()
    sf.SetSample(0.5, samplemode = 0, samplestate = 0)
    sf.AddPotentialFeatures(['Pclass'])
    sf.clf = LogisticRegression()
    sf.SetLogFile('record2.log')
#    sf.SetFeaturesLimit(5)
    sf.SetTimeLimit(0.2)
    sf.run(validation)

if __name__ == "__main__":
    main()
