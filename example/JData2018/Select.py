import lightgbm as lgbm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
from MLFeatureSelection import FeatureSelection as FS

def score1(pred, real):
    pred['index'] = np.arange(pred.shape[0]) + 1
    pred['wi'] = 1 / (1 + np.log(pred['index']))

    compare = pd.merge(pred, real, how='left', on='user_id')
    compare = compare[:50000]
    compare.fillna(0, inplace=True)
    S1 = np.sum(compare['buy'] * compare['wi']) / compare['wi'].sum()
    return S1

def prepareData():
    return pd.read_csv('train_demo.csv')

def validate(X, y, features, clf, score):
    Performance = []
    for day in [335]:
        Ttrain = X.CreateGroup < day
        Ttest = X.CreateGroup == day
        Testtemp = X[Ttest]
        X_train, X_test = X[Ttrain], X[Ttest]
        X_train, X_test = X_train[features], X_test[features]
        y_train, y_test = y[Ttrain], y[Ttest]
        clf.fit(X_train,y_train, eval_set = [(X_train, y_train), (X_test, y_test)], eval_metric='auc', verbose=False,early_stopping_rounds=200)
        Testtemp['Prob'] = clf.predict_proba(X_test)[:,1]
        Testtemp['Days'] = 0
        prediction = Testtemp[['user_id','Prob','Days']]
        prediction.sort_values(by = ['Prob'], ascending = False, inplace = True)
        Performance.append(score(prediction[['user_id','Days']], Testtemp[['user_id', 'nextbuy','buy']]))
    print("Mean Score: {}".format(np.mean(Performance)))
    return np.mean(Performance)

def main():
    sf = FS.Select(Sequence = True, Random = False, Cross = False)
    sf.ImportDF(prepareData(),label = 'buy')
    sf.ImportLossFunction(score1, direction = 'ascend')
    #sf.ImportCrossMethod(CrossMethod)
    sf.InitialNonTrainableFeatures(['buy','nextbuy','o_date','a_date','PredictDays','user_id'])
    sf.InitialFeatures(['age_x', 'sex_x', 'user_lv_cd_x', 'buycnt', 'daybeforelastbuy_o_ave'])
    #sf.PotentialAdd = ['daybeforelastbuy_o_mean']
    #sf.clf = lgbm.LGBMClassifier(random_state=1, num_leaves = 6, n_estimators=5000, max_depth=3, learning_rate = 0.05, n_jobs=1)
    sf.clf = lgbm.LGBMClassifier(random_state=1, num_leaves = 6, n_estimators=1000, max_depth=3, learning_rate = 0.2, n_jobs=8)
    sf.SetLogFile('record.log')
    sf.run(validate)

if __name__ == "__main__":
    main()
