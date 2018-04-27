#!/usr/bin/env python
#-*- coding:utf-8 -*-

##############################################
# File Name: FeatureSelection
# Author: Xuhao Du
# Email: duxuhao88@gmail.com
##############################################

from scipy.stats import pearsonr
from collections import OrderedDict
import random
import numpy as np
from sklearn.model_selection import KFold

def DefaultValidation(X, y, clf, lossfunction):
    totaltest = []
    kf = KFold(5)
    for train_index, test_index in kf.split(self.df):
        X_train, X_test = self.df.ix[train_index,:][self.features], self.df.ix[test_index,:][self.features]
        y_train, y_test = self.df.ix[train_index,:].Label, self.df.ix[test_index,:].Label
        clf.fit(X_train, y_train)
        totaltest.append(lossfunction(y_test, clf.predict_proba(X_test)[:,1]))
    return np.mean(totaltest)

class LRS_SA_RGSS_combination():
    def __init__(self, clf, df, RecordFolder, columnname, start, label,  Process, direction, LossFunction, validatefunction = 0, Startcol = ['None'], PotentialAdd = [], CrossMethod = 0, CoherenceThreshold = 1):
        self.clf = clf
        self.LossFunction = LossFunction
        self.df = df
        self.RecordFolder  = RecordFolder
        self.columnname = columnname
        self.TemplUsedFeatures, self.Label = start, label
        self.PotentialAdd = PotentialAdd # you need to add some potential feature here, otherwise the Simulate Anneal Arithmetic will not work
        self.Startcol = Startcol
        self.CrossMethod = CrossMethod
        self.Process = Process
        self.direction = direction
        self.validatefunction = validatefunction
        if self.validatefunction == 0:
            self.validatefunction = DefaultValidation # DefaultValidation is 5-fold
        self.coherenceThreshold = CoherenceThreshold

    def evaluate(self, a, b):
        if self.direction == 'ascend':
            return a > b
        else:
            return a < b

    def select(self):
        #change them based on your evaluation function,
        #if smaller the better, self.score, self.greedyscore = 1, 0
        #if larger the better, self.score, self.greedyscore = 0, 1
        if self.direction == 'ascend':
            self.score, self.greedyscore = 0, 1
        else:
            self.score, self.greedyscore = 1, 0
        self.remain = '' # for initial
        while self.evaluate(self.greedyscore, self.score) | self.ScoreUpdate():
            #if the random selection have a better combination,
            #the greedy will loop again. otherwise, the selection complete
            print('test performance of initial features combination')
            self.bestscore, self.bestfeature = self.score, self.TemplUsedFeatures[:]
            self.validation(self.TemplUsedFeatures[:], str(0), 'baseline', coetest = 0)
            # greedy: forward + backward + Simulated Annealing
            if self.Process[0]:
                self.Greedy()
            self.ScoreUpdate()
            self.greedyscore = self.bestscore
            print('random select starts with:\n {0}\n score: {1}'.format(self.bestfeature, self.greedyscore))
            # random selection
            if self.Process[1]:
                self.MyRandom()

            if self.Process[2]:
                if 1: #self.greedyscore == self.score:
                    print('small cycle cross')
                    n = 1
                    while self.ScoreUpdate() | n:
                        # only if the small cycle cross can construct better features,
                        # then start next small round, otherwise, go to medium cycle
                        self.CrossTermSearch(self.bestfeature, self.bestfeature)
                        n = 0
                if self.greedyscore == self.score:
                    print('medium cycle cross')
                    n = 1
                    while self.ScoreUpdate() | n:
                        # only if the medium cycle cross can construct better features,
                        # then start next medium round, otherwise, go to large cycle
                        self.CrossTermSearch(self.columnname, self.bestfeature)
                        n = 0
                if self.greedyscore == self.score:
                    print('large cycle cross')
                    n = 1
                    while self.ScoreUpdate() | n:
                        # only if the medium cycle cross can construct better features,
                        # then start next medium round, otherwise, go to large cycle
                        self.CrossTermSearch(self.columnname, self.columnname)
                        n = 0
            self.ScoreUpdate()
        print('{0}\nbest score:{1}\nbest features combination: {2}'.format('*-*' * 50, self.bestscore, self.bestfeature))

    def validation(self, selectcol, num, addfeature, coetest=0): #get the score with the new features list and update the best features combination
        """ set up your cross validation here"""
        selectcol = list(OrderedDict.fromkeys(selectcol))
        X, y = self.df, self.df[self.Label]
        totaltest = self.validatefunction(X[selectcol], y, self.clf, self.LossFunction)
        print('Mean loss: {}'.format(totaltest))
        # only when the score improve, the program will record,
        # change the operator ( < or > ) according to your evalulation function
        if self.evaluate(np.mean(totaltest), self.score):
            cc = [0]
            if self.coherenceThreshold != 1: #execute in the features adding process
                coltemp = selectcol[:]
                coltemp.remove(addfeature)
                cc = [pearsonr(self.df[addfeature],self.df[ct])[0] for ct in coltemp] #check the correlation coefficient
            # to see the correlation coefficient between each two features,
            # not select the feature if its correlation coefficient is too high
            if (np.abs(np.max(cc)) < self.coherenceThreshold):
                with open(self.RecordFolder, 'a') as f: #record all the imporved combination
                    f.write('{0}  {1}  {2}:\n{3}\t{4}\n'.format(num, addfeature, np.abs(np.max(cc)), np.round(np.mean(totaltest),6), selectcol[:], '*-' * 50))
                self.TemplUsedFeatures, self.score = selectcol[:], np.mean(totaltest)
                if num == 'reverse':
                    self.dele = addfeature
                else:
                    self.remain = addfeature # updaet the performance

    def Greedy(self):
        col = self.columnname[:]
        print('{0}{1}{2}'.format('-' * 20, 'start greedy', '-' * 20))
        for i in self.TemplUsedFeatures:
            print(i)
            try:
                col.remove(i)
            except:
                pass
        self.dele = ''
        self.bestscore, self.bestfeature = self.score, self.TemplUsedFeatures[:]
        while (self.Startcol != self.TemplUsedFeatures) | (self.PotentialAdd != []): #stop when no improve for the last round and no potential add feature
            if self.Startcol == self.TemplUsedFeatures:
                self.ScoreUpdate()
                self.score += 0.001 #Simulate Anneal Arithmetic, step back a bit, the value need to be change
                self.TemplUsedFeatures.append(self.PotentialAdd[0])
            print('{0} {1} round {2}'.format('*' * 20, len(self.TemplUsedFeatures)+1, '*' * 20))
            if self.remain in col:
                col.remove(self.remain)
            if self.dele != '':
                col.append(self.dele)
            self.Startcol = self.TemplUsedFeatures[:]
            for sub, i in enumerate(col): #forward sequence selection add one each round
                print(i)
                print('{}/{}'.format(sub,len(col)))
                selectcol = self.Startcol[:]
                selectcol.append(i)
                self.validation(selectcol, str(1+sub), i, coetest = 0)
            for sr, i in enumerate(self.TemplUsedFeatures[:-1]): # backward sequence selection, -2 becuase the last 2 is just selected
                deletecol = self.TemplUsedFeatures[:] # can delete several each round
                if i in deletecol:
                    deletecol.remove(i)
                print(i)
                print('reverse {}/{}'.format(sr,len(self.TemplUsedFeatures[:-1])))
                self.validation(deletecol, 'reverse', i, coetest = 0)
            for i in self.TemplUsedFeatures:
                if i in self.PotentialAdd:
                    self.PotentialAdd.remove(i)
        print('{0}{1}{2}'.format('-' * 20, 'complete greedy', '-' * 20))

    def MyRandom(self):
        self.ScoreUpdate()
        col = self.columnname[:]
        print('{0}{1}{2}'.format('-' * 20, 'start random', '-' * 20))
        for i in self.bestfeature:
            col.remove(i)
        for t in range(3,8): # add 4 to 8 features randomly, choose your own range
            print('add {} features'.format(t))
            for i in range(50): # run 50 rounds each quantity, choose your own round number
                selectcol = random.sample(col, t)
                recordadd = selectcol[:]
                for add in self.bestfeature:
                    selectcol.append(add)
                self.validation(selectcol, str(i), str(recordadd))
        print('{0}{1}{2}'.format('-' * 20, 'complete random', '-' * 20))

    def ScoreUpdate(self):
        if self.direction == 'ascend':
            start = 0
        else:
            start = 1
        if self.score == start:
            return True
        elif self.evaluate(self.score, self.bestscore):
            self.bestscore, self.bestfeature = self.score, self.TemplUsedFeatures[:]
            return True
        return False

    def CrossTermSearch(self, col1, col2):
        self.ScoreUpdate()
        Effective = []
        crosscount = 0
        for c1 in col1:
            for c2 in col2[::-1]:
                for oper in self.CrossMethod.keys():
                    print('{}/{}'.format(crosscount, len(col1) * len(col2[::-1])))
                    crosscount += 1
                    newcolname = "({}{}{})".format(c1,oper,c2)
                    self.df[newcolname] = self.CrossMethod[oper](self.df[c1], self.df[c2])
                    selectcol = self.bestfeature[:]
                    selectcol.append(newcolname)
                    try:
                        self.validation(selectcol, 'cross term', newcolname, coetest = 0)
                    except:
                        pass
                    if self.ScoreUpdate():
                        Effective.append(newcolname)
                    else:
                        self.df.drop(newcolname, axis = 1,inplace=True)
        Effective.remove(self.remain)
        for rm in Effective:
             self.df.drop(rm, axis = 1, inplace=True)
        self.columnname.append(self.remain)

class Select():
    def __init__(self, Sequence = True, Random = True, Cross = True):
        self.Sequence = Sequence
        self.Random = Random
        self.Cross = Cross
        self.NonTrainableFeatures = []
        self.temp = []
        self.logfile = 'record.log'
        self.PotentialAdd = []
        self.CrossMethod = 0

    def ImportCrossMethod(self, CrossMethod):
        self.CrossMethod = CrossMethod

    def ImportDF(self, df, label):
        self.df = df
        self.label = label

    def ImportLossFunction(self, modelscore, direction):
        self.modelscore = modelscore
        self.direction = direction

    def InitialFeatures(self,features):
        self.temp = features

    def obtaincol(self):
        """ for getting rid of the useless columns in the dataset
        """
        self.ColumnName = list(self.df.columns)
        for i in self.NonTrainableFeatures:
            if i in self.ColumnName:
                self.ColumnName.remove(i)
        return self.ColumnName

    def run(self,validate):
        self.obtaincol()
        # start selecting
        with open(self.logfile, 'a') as f:
            f.write('\n{}\n%{}%\n'.format('Start!','-'*60))
        a = LRS_SA_RGSS_combination(df = self.df,
                                    clf = self.clf,
                                    RecordFolder = self.logfile,
                                    LossFunction = self.modelscore,
                                    label = self.label,
                                    columnname = self.ColumnName[:],
                                    start = self.temp,
                                    CrossMethod = self.CrossMethod, # your cross term method
                                    PotentialAdd = self.PotentialAdd, # potential feature for Simulated Annealing
                                    Process = [self.Sequence, self.Random, self.Cross],
                                    direction = self.direction,
                                    validatefunction = validate,
                                    )
        try:
            a.select()
        finally:
            with open(self.logfile, 'a') as f:
                f.write('\n{}\n%{}%\n'.format(self.temp,'-'*60))
