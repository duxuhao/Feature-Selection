import lightgbm as lgbm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
from MLFeatureSelection import sequence_selection as ss

def score(pred, real): #评分系统，感谢herhert，返回s2
    pred['index'] = np.arange(pred.shape[0]) + 1
    pred['wi'] = 1 / (1 + np.log(pred['index']))

    compare = pd.merge(pred, real, how='left', on='user_id')
    compare = compare[:50000]
    compare.fillna(0, inplace=True)
    S1 = np.sum(compare['buy'] * compare['wi']) / compare['wi'].sum()

    compare_for_S2 = compare[compare['buy'] == 1]
    S2 = np.sum(10 / (10 + np.square(compare_for_S2['Days'] - compare_for_S2['nextbuy']))) / real.shape[0]

    S = 0.4 * S1 + 0.6 * S2
    print("S1=", S1, "| S2 ", S2)
    print("S =", S)
    return S2

def prepareData():
    return pd.read_csv('train_demo.csv') #读入你自己的数据集

df = prepareData() #获得数据集
clf0 = lgbm.LGBMClassifier(random_state=1, num_leaves = 6, n_estimators=1000, max_depth=3, learning_rate = 0.2, n_jobs=8) #先固定分类模型
features1 = ['age_x', 'sex_x', 'user_lv_cd_x', 'buycnt', 'daybeforelastbuy_o_ave', 'daybeforelastbuy_o_sum', 'daybeforelastbuy_o_median', 'daybeforelastbuy_o_min', 'daybeforelastbuy_o_max', 'daybeforelastbuy']

Ttrain = df.CreateGroup < 335 #我的数据集划分，335指的是第335天，其实也就是前11个月
X_train = df[Ttrain][features1]
y_train = df[Ttrain].buy
clf0.fit(X_train,y_train) #获得分类模型

def validate(X, y, features, clf, score): #测评系统
    Performance = []
    for day in [335]:
        Ttrain = X.CreateGroup < day
        Ttest = X.CreateGroup == day
        Testtemp = X[Ttest]
        X_train, X_test = X[Ttrain], X[Ttest]
        X_train, X_test = X_train[features1], X_test[features1]
        Testtemp['Prob'] = clf0.predict_proba(X_test)[:,1] #得到所有小伙伴购买的概率
        X_train, X_test = X[Ttrain], X[Ttest]
        X_train, X_test = X_train[features], X_test[features] #回归用到的训练集和测试集
        y_train, y_test = y[Ttrain], y[Ttest]
        clf.fit(X_train,y_train)
        Testtemp['Days'] = clf.predict(X_test)
        prediction = Testtemp[['user_id','Prob','Days']]
        prediction.sort_values(by = ['Prob'], ascending = False, inplace = True) #排序
        Performance.append(score(prediction[['user_id','Days']], Testtemp[['user_id', 'nextbuy','buy']]))
    print("Mean Score: {}".format(np.mean(Performance)))
    return np.mean(Performance), clf #返回分数

def main():
    sf = ss.Select(Sequence = True, Random = False, Cross = False) #初始化选择器，选择你需要的流程
    sf.ImportDF(df,label = 'nextbuy') #导入数据集以及目标标签
    sf.ImportLossFunction(score, direction = 'ascend') #导入评价函数以及优化方向
    sf.InitialNonTrainableFeatures(['buy','nextbuy','o_date','a_date','PredictDays','user_id']) #初始化不能用的特征
    sf.InitialFeatures(['age_x', 'sex_x', 'user_lv_cd_x', 'buycnt', 'daybeforelastbuy_o_ave']) #初始化其实特征组合
    sf.GenerateCol() #生成特征库 （具体该函数变量请参考根目录下的readme）
    sf.SetSample(1, samplemode = 1) #初始化抽样比例和随机过程
    sf.SetTimeLimit(100) #设置算法运行最长时间，以分钟为单位
    sf.clf = lgbm.LGBMRegressor(random_state=1, num_leaves =6, n_estimators=1000, max_depth=3, learning_rate = 0.2, n_jobs=8) #设定回归模型
    sf.SetLogFile('record.log') #初始化日志文件
    sf.run(validate) #输入检验函数并开始运行

if __name__ == "__main__":
    main()
