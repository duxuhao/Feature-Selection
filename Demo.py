import lightgbm as lgbm
import pandas as pd
import numpy as np
import datetime
from MLFeatureSelection import sequence_selection, importance_selection, coherence_selection,tools

def read():
    train = pd.read_csv('traina_8month_more.csv')
    pre = pd.read_csv('testa_8month_more.csv')
    df = pd.concat([train, pre]).reset_index(drop=True)
    return df

def score1(pred, real):
    pred['index'] = np.arange(pred.shape[0]) + 1
    pred['wi'] = 1 / (1 + np.log(pred['index']))

    compare = pd.merge(pred, real, how='left', on='user_id')
    compare = compare[:34000]
    compare.fillna(0, inplace=True)
    S1 = np.sum(compare['buy'] * compare['wi']) / compare['wi'].sum()
    return S1

def validate(X_all, y, features, clf, score):
    for day in [335]:
        Ttrain = X.CreateGroup < day
        Ttest = (X.CreateGroup == day)
        Testtemp = X[Ttest]
        X_train, X_test = X[Ttrain], X[Ttest]
        X_train, X_test = X_train[features], X_test[features]
        y_train, y_test = X[Ttrain].buy, X[Ttest].buy
        clf.fit(X_train,y_train, eval_set = [(X_train, y_train), (X_test, y_test)], eval_metric='auc', verbose=False, early_stopping_rounds=50)
        Testtemp['Prob'] = clf.predict_proba(X_test)[:,1]
        Testtemp['Days'] = 7
        prediction = Testtemp[['user_id','Prob','Days']]
        prediction.sort_values(by = ['Prob'], ascending = False, inplace = True)
        Performance = score(prediction[['user_id','Days']], Testtemp[['user_id', 'nextbuy','buy']])
    print("Mean Score: {}".format(Performance))
    return Performance,clf

def seq(df,f, notusable,clf):
    sf = sequence_selection.Select(Sequence = True, Random = True, Cross = True) #初始化选择器，选择你需要的流程
    sf.ImportDF(df,label = 'buy') #导入数据集以及目标标签
    sf.ImportLossFunction(score1, direction = 'ascend') #导入评价函数以及优化方向
    #sf.ImportCrossMethod(CrossMethod)
    sf.InitialNonTrainableFeatures(notusable) #初始化不能用的特征
    sf.InitialFeatures(f)
    sf.GenerateCol() #生成特征库 （具体该函数变量请参考根目录下的readme）
#    sf.SetTimeLimit(120) #设置算法运行最长时间，以分钟为单位
    sf.clf = clf
    sf.SetLogFile('record_seq8.log') #初始化日志文件
    return sf.run(validate) #输入检验函数并开始运行

def imp(df,f,clf):
    sf = importance_selection.Select() 
    sf.ImportDF(df,label = 'buy') #import dataset
    sf.ImportLossFunction(score1, direction = 'ascend') 
    sf.InitialFeatures(f)
    sf.SelectRemoveMode(batch = 5)
    sf.clf = clf
    sf.SetLogFile('record_imp8.log')
    return sf.run(validate)

def coh(df,f,clf):
    sf = coherence_selection.Select() 
    sf.ImportDF(df,label = 'buy')
    sf.ImportLossFunction(score1, direction = 'ascend') 
    sf.InitialFeatures(f)
    sf.SelectRemoveMode(batch=5, lowerbound = 0.9)
    sf.clf = clf 
    sf.SetLogFile('record_coh8.log') 
    return sf.run(validate)

if __name__ == "__main__":
    df = read1()
    notusable = ['buy','nextbuy','o_date','a_date','PredictDays','user_id']
        
    f = tools.readlog('record.log',0.645769)
    clf = xgb.XGBClassifier()
    n = 1
    uf = f[:]
    while n | (uf != f):
        n = 0
        print('importance selection')
        uf = imp(df,uf,clf)
        print(uf)
        print('coherence selection')
        uf = coh(df,uf,clf)
        print('sequence selection')
        seq(df, uf, notusable,clf)