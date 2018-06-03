import lightgbm as lgbm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
from MLFeatureSelection import sequence_selection as ss

def score1(pred, real): #评分系统，感谢herhert，只要s1
    pred['index'] = np.arange(pred.shape[0]) + 1
    pred['wi'] = 1 / (1 + np.log(pred['index']))

    compare = pd.merge(pred, real, how='left', on='user_id')
    compare = compare[:50000]
    compare.fillna(0, inplace=True)
    S1 = np.sum(compare['buy'] * compare['wi']) / compare['wi'].sum()
    return S1

def prepareData(): #读入你自己的数据集
    return pd.read_csv('train_demo.csv')

def validate(X, y, features, clf, score):
    Performance = []
    for day in [335]: #我的数据集划分，335指的是第335天，其实也就是前11个月
        Ttrain = X.CreateGroup < day
        Ttest = X.CreateGroup == day
        Testtemp = X[Ttest]
        X_train, X_test = X[Ttrain], X[Ttest]
        X_train, X_test = X_train[features], X_test[features]
        y_train, y_test = y[Ttrain], y[Ttest]
        clf.fit(X_train,y_train, eval_set = [(X_train, y_train), (X_test, y_test)], eval_metric='auc', verbose=False,early_stopping_rounds=200)
        Testtemp['Prob'] = clf.predict_proba(X_test)[:,1]
        Testtemp['Days'] = 0 #这是预测日期结果，这里只优化s1
        prediction = Testtemp[['user_id','Prob','Days']]
        prediction.sort_values(by = ['Prob'], ascending = False, inplace = True)
        Performance.append(score(prediction[['user_id','Days']], Testtemp[['user_id', 'nextbuy','buy']]))
    print("Mean Score: {}".format(np.mean(Performance)))
    return np.mean(Performance),clf

def main():
    sf = sequence_selection as ss.Select(Sequence = True, Random = False, Cross = False) #初始化选择器，选择你需要的流程
    sf.ImportDF(prepareData(),label = 'buy') #导入数据集以及目标标签
    sf.ImportLossFunction(score1, direction = 'ascend') #导入评价函数以及优化方向
    #sf.ImportCrossMethod(CrossMethod)
    sf.InitialNonTrainableFeatures(['buy','nextbuy','o_date','a_date','PredictDays','user_id']) #初始化不能用的特征
    sf.InitialFeatures(['age_x', 'sex_x', 'user_lv_cd_x', 'buycnt', 'daybeforelastbuy_o_ave']) #初始化其实特征组合
    sf.GenerateCol() #生成特征库 （具体该函数变量请参考根目录下的readme）
    sf.SetSample(1, samplemode = 1) #初始化抽样比例和随机过程
    sf.SetTimeLimit(100) #设置算法运行最长时间，以分钟为单位
    sf.clf = lgbm.LGBMClassifier(random_state=1, num_leaves = 6, n_estimators=1000, max_depth=3, learning_rate = 0.2, n_jobs=8) #设定回归模型
    sf.SetLogFile('record.log') #初始化日志文件
    sf.run(validate) #输入检验函数并开始运行

if __name__ == "__main__":
    main()
