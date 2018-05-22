#!/usr/bin/env python
#-*- coding:utf-8 -*-

##############################################
# File Name: sequence_selection.py
# Author: Xuhao Du
# Email: duxuhao88@gmail.com
##############################################

from scipy.stats import pearsonr
from collections import OrderedDict
import random
import time
import numpy as np
from sklearn.model_selection import KFold
import sys

def DefaultValidation(X, y, features, clf, lossfunction, fit_params=None):
    totaltest = []
    kf = KFold(5)
    for train_index, test_index in kf.split(X):
        X_train, X_test = X.ix[train_index,:][features], X.ix[test_index,:][features]
        y_train, y_test = y.ix[train_index,:].Label, y.ix[test_index,:].Label
        clf.fit(X_train, y_train, **fit_params)
        totaltest.append(lossfunction(y_test, clf.predict_proba(X_test)[:,1]))
    return np.mean(totaltest)

def _reachlimit(func):
    def wrapper(c):
        temp = func(c)
        if (len(c._TemplUsedFeatures) >= c._FeaturesQuanLimitation) | ((time.time() - c._StartTime) >= c._TimeLimitation):
            if (len(c._TemplUsedFeatures) >= c._FeaturesQuanLimitation):
                print("Maximum features limit reach!")
            if ((time.time() - c._StartTime) >= c._TimeLimitation):
                print("Time's up!")
            print('{0}\nbest score:{1}\nbest {2} features combination: {3}'.format('*-*' * 50,
                                                                                   c._score,
                                                                                   c._FeaturesQuanLimitation,
                                                                                   c._TemplUsedFeatures))
            sys.exit()
        return temp
    return wrapper

class _LRS_SA_RGSS_combination(object):

    def __init__(self, clf, df, RecordFolder, columnname, start, label,
                 Process, direction, LossFunction, FeaturesQuanLimitation,
                 TimeLimitation, SampleRatio=1, SampleMode=1, SampleState=0,
                 fit_params=None, validatefunction=0,
                 PotentialAdd=[], CrossMethod=0, CoherenceThreshold=1):
        self._clf = clf
        self._fit_params = fit_params
        self._LossFunction = LossFunction
        self._df = df
        self._RecordFolder = RecordFolder
        self._columnname = columnname
        self._TemplUsedFeatures, self._Label = start, label
        self._PotentialAdd = PotentialAdd # you need to add some potential feature here, otherwise the Simulate Anneal Arithmetic will not work
        self._Startcol = ['None']
        self._CrossMethod = CrossMethod
        self.Process = Process
        self._direction = direction
        self._validatefunction = validatefunction
        if self._validatefunction == 0:
            self._validatefunction = DefaultValidation # DefaultValidation is 5-fold
        self._coherenceThreshold = CoherenceThreshold
        self._TimeLimitation = TimeLimitation * 60
        self._FeaturesQuanLimitation = FeaturesQuanLimitation
        self._bestfeature = self._TemplUsedFeatures[:]
        self._StartTime = time.time()
        self._fit_params = fit_params
        if SampleRatio > 1:
            self._sampleratio = 1
        elif SampleRatio <= 0:
            print("sample ratio should be positive, the set up sample ratio is wrong")
            sys.exit()
        else:
            self._sampleratio = SampleRatio
        self._samplestate = SampleState
        self._samplemode = SampleMode

    def _evaluate(self, a, b):
        if self._direction == 'ascend':
            return a > b
        else:
            return a < b

    def select(self):
        self._StartTime = time.time()
        #change them based on your evaluation function,
        #if smaller the better, self._score, self._greedyscore = 1, 0
        #if larger the better, self._score, self._greedyscore = 0, 1
        if self._direction == 'ascend':
            self._score, self._greedyscore = -np.inf, np.inf
        else:
            self._score, self._greedyscore = np.inf, -np.inf
        self.remain = '' # for initial
        self._first = 1
        while self._evaluate(self._score, self._greedyscore) | self._first:
            #if the random selection have a better combination,
            #the greedy will loop again. otherwise, the selection complete
            print('test performance of initial features combination')
            self.bestscore, self._bestfeature = self._score, self._TemplUsedFeatures[:]
            if (self._TemplUsedFeatures[:] != []) & (self._first == 1):
                self._validation(self._TemplUsedFeatures[:],
                                 str(0), 'baseline', coetest = 0)
            # greedy: forward + backward + Simulated Annealing
            if self.Process[0]:
                self._Greedy()
            self._ScoreUpdate()
            self._greedyscore = self.bestscore
            print('random select starts with:\n {0}\n score: {1}'.format(self._bestfeature,
                                                                         self._greedyscore))
            # random selection
            if self.Process[1]:
                self._MyRandom()

            if self.Process[2] & (self._first == 1): # avoid cross term twice until it is fix
                if 1: #self._greedyscore == self._score:
                    print('small cycle cross')
                    n = 1
                    while self._ScoreUpdate() | n:
                        # only if the small cycle cross can construct better features,
                        # then start next small round, otherwise, go to medium cycle
                        self._CrossTermSearch(self._bestfeature, self._bestfeature)
                        n = 0
                if self._greedyscore == self._score:
                    print('medium cycle cross')
                    n = 1
                    while self._ScoreUpdate() | n:
                        # only if the medium cycle cross can construct better features,
                        # then start next medium round, otherwise, go to large cycle
                        self._CrossTermSearch(self._columnname, self._bestfeature)
                        n = 0
                if self._greedyscore == self._score:
                    print('large cycle cross')
                    n = 1
                    while self._ScoreUpdate() | n:
                        # only if the medium cycle cross can construct better features,
                        # then start next medium round, otherwise, go to large cycle
                        self._CrossTermSearch(self._columnname, self._columnname)
                        n = 0
            self._first = 0
            self._ScoreUpdate()
        print('{0}\nbest score:{1}\nbest features combination: {2}'.format('*-*' * 50,
                                                                           self.bestscore,
                                                                           self._bestfeature))
        with open(self._RecordFolder, 'a') as f:
            f.write('{0}\nbest score:{1}\nbest features combination: {2}'.format('*-*' * 50,
                                                                           self.bestscore,
                                                                           self._bestfeature))

    def _validation(self,
                    selectcol,
                    num,
                    addfeature,
                    coetest=0): #get the score with the new features list and update the best features combination
        """ set up your cross validation here"""
        self.chekcLimit()
        selectcol = list(OrderedDict.fromkeys(selectcol))
        self._samplestate += self._samplemode
        if self._sampleratio < 1:
            tempdf = self._df.sample(frac = self._sampleratio, random_state = self._samplestate).reset_index(drop = True)
        else:
            tempdf = self._df
        #X, y = self._df, self._df[self._Label]
        X, y = tempdf, tempdf[self._Label]
        totaltest = self._validatefunction(X, y, selectcol,
                                           self._clf, self._LossFunction) #, self._fit_params)
        print('Mean loss: {}'.format(totaltest))
        # only when the score improve, the program will record,
        # change the operator ( < or > ) according to your evalulation function
        if self._evaluate(np.mean(totaltest), self._score):
            cc = [0]
            if self._coherenceThreshold != 1: #execute in the features adding process
                coltemp = selectcol[:]
                coltemp.remove(addfeature)
                cc = [pearsonr(self._df[addfeature],self._df[ct])[0] for ct in coltemp] #check the correlation coefficient
            # to see the correlation coefficient between each two features,
            # not select the feature if its correlation coefficient is too high
            if (np.abs(np.max(cc)) < self._coherenceThreshold):
                with open(self._RecordFolder, 'a') as f: #record all the imporved combination
                    f.write('{0}  {1}  {2}:\n{3}\t{4}\n'.format(num, addfeature,
                                                                np.abs(np.max(cc)),
                                                                np.round(np.mean(totaltest),6),
                                                                selectcol[:], '*-' * 50))
                self._TemplUsedFeatures, self._score = selectcol[:], np.mean(totaltest)
                if num == 'reverse':
                    self.dele = addfeature
                else:
                    self.remain = addfeature # updaet the performance

    def _Greedy(self):
        col = self._columnname[:]
        print('{0}{1}{2}'.format('-' * 20, 'start greedy', '-' * 20))
        for i in self._TemplUsedFeatures:
            print(i)
            try:
                col.remove(i)
            except:
                pass
        self.dele = ''
        self.bestscore, self._bestfeature = self._score, self._TemplUsedFeatures[:]
        while (self._Startcol != self._TemplUsedFeatures) | (self._PotentialAdd != []): #stop when no improve for the last round and no potential add feature
            if self._Startcol == self._TemplUsedFeatures:
                self._ScoreUpdate()
                if self._direction == 'ascend':
                    self._score *= 0.95 #Simulate Anneal Arithmetic, step back a bit, the value need to be change
                else:
                    self._score /= 0.95
                self._TemplUsedFeatures.append(self._PotentialAdd[0])
            print('{0} {1} round {2}'.format('*' * 20, len(self._TemplUsedFeatures)+1, '*' * 20))
            if self.remain in col:
                col.remove(self.remain)
            if self.dele != '':
                col.append(self.dele)
            self._Startcol = self._TemplUsedFeatures[:]
            for sub, i in enumerate(col): #forward sequence selection add one each round
                print(i)
                print('{}/{}'.format(sub,len(col)))
                selectcol = self._Startcol[:]
                selectcol.append(i)
                self._validation(selectcol, str(1+sub), i, coetest = 0)
            for sr, i in enumerate(self._TemplUsedFeatures[:-1]): # backward sequence selection, -2 becuase the last 2 is just selected
                deletecol = self._TemplUsedFeatures[:] # can delete several each round
                if i in deletecol:
                    deletecol.remove(i)
                print(i)
                print('reverse {}/{}'.format(sr,len(self._TemplUsedFeatures[:-1])))
                self._validation(deletecol, 'reverse', i, coetest = 0)
            for i in self._TemplUsedFeatures:
                if i in self._PotentialAdd:
                    self._PotentialAdd.remove(i)
        print('{0}{1}{2}'.format('-' * 20, 'complete greedy', '-' * 20))

    def _MyRandom(self):
        self._ScoreUpdate()
        col = self._columnname[:]
        print('{0}{1}{2}'.format('-' * 20, 'start random', '-' * 20))
        for i in self._bestfeature:
            if i in col:
                col.remove(i)
        random.seed(a = self._samplestate)
        for t in range(3,8): # add 4 to 8 features randomly, choose your own range
            if t < len(col):
                print('add {} features'.format(t))
                for i in range(50): # run 50 rounds each quantity, choose your own round number
                    selectcol = random.sample(col, t)
                    recordadd = selectcol[:]
                    for add in self._bestfeature:
                        selectcol.append(add)
                    self._validation(selectcol, str(i), str(recordadd))
        print('{0}{1}{2}'.format('-' * 20, 'complete random', '-' * 20))

    @_reachlimit
    def chekcLimit(self):
        return True

    @_reachlimit
    def _ScoreUpdate(self):
        if self._direction == 'ascend':
            start = -np.inf
        else:
            start = np.inf
        if self._score == start:
            return True
        elif self._evaluate(self._score, self.bestscore):
            self.bestscore, self._bestfeature = self._score, self._TemplUsedFeatures[:]
            return True
        return False

    def _CrossTermSearch(self, col1, col2):
        self._ScoreUpdate()
        Effective = []
        crosscount = 0
        for c1 in col1:
            for c2 in col2[::-1]:
                for oper in self._CrossMethod.keys():
                    print('{}/{}'.format(crosscount, len(self._CrossMethod.keys()) * len(col1) * len(col2[::-1])))
                    crosscount += 1
                    newcolname = "({}{}{})".format(c1,oper,c2)
                    self._df[newcolname] = self._CrossMethod[oper](self._df[c1], self._df[c2])
                    selectcol = self._bestfeature[:]
                    selectcol.append(newcolname)
                    try:
                        self._validation(selectcol, 'cross term', newcolname, coetest = 0)
                    except:
                        pass
                    if self._ScoreUpdate():
                        Effective.append(newcolname)
                    else:
                        self._df.drop(newcolname, axis = 1, inplace=True)
        Effective.remove(self.remain)
#        for rm in Effective:
#             self._df.drop(rm, axis = 1, inplace=True)
        self._columnname.append(self.remain)

class Select(object):
    """This is a class for sequence/random/crossterm features selection

    The functions needed to be called before running include:

        ImportDF(pd.dataframe, str) - import you complete dataset including the label column

        ImportLossFunction(func, str) - import your self define loss function,
                                        eq. logloss, accuracy, etc

        InitialFeatures(list) - Initial your starting features combination,
                                if the initial features combination include
                                all features, the backward sequence searching
                                will run automatically

        InitialNonTrainableFeatures(list) - Initial the non-trainable features

        ImportCrossMethod(dict) - Import your cross method, eq. +, -, *, /,
                                  can be as complicate as you want, this requires
                                  setup if Cross = True

        AddPotentialFeatures(list) - give some strong features you think might
                                     be useful. It will force to add into the
                                     features combination once the sequence
                                     searching doesn't improve

        SetCCThreshold(float) - Set the maximum correlation coefficient between each
                                features

        run(func) - start selecting features
    """

    def __init__(self, Sequence=True, Random=True, Cross=True):
        self.Sequence = Sequence
        self.Random = Random
        self.Cross = Cross
        self._NonTrainableFeatures = []
        self._temp = []
        self._logfile = 'record.log'
        self._PotentialAdd = []
        self._CrossMethod = 0
        self._CoherenceThreshold = 1
        self._FeaturesLimit = np.inf
        self._TimeLimit = np.inf
        self._sampleratio = 1
        self._samplestate = 0
        self._samplemode = 1

    def SetLogFile(self, fn):
        """Setup the log file

        Args:
            fn: str, filename
        """
        self._logfile = fn

    def ImportDF(self, df, label):
        """Import pandas dataframe to the class

        Args:
            df: pandas dataframe include all features and label.
            label: str, label name
        """
        self._df = df
        self._label = label

    def ImportCrossMethod(self, CrossMethod):
        """Import a dictionary with different cross function

        Args:
            CrossMethod: dict, dictionary with different cross function
        """
        self._CrossMethod = CrossMethod

    def ImportLossFunction(self, modelscore, direction):
        """Import the loss function

        Args:
            modelscore: the function to calculate the loss result
                        with two input series
            direction: str, ‘ascent’ or descent, the way you want
                       the score to go
        """
        self._modelscore = modelscore
        self._direction = direction

    def InitialFeatures(self,features):
        """Initial your starting features combination

        Args:
            features: list, the starting features combination
        """
        self._temp = features

    def InitialNonTrainableFeatures(self, features):
        """Setting the nontrainable features, eq. user_id

        Args:
            features: list, the nontrainable features
        """
        self._NonTrainableFeatures = features

    def GenerateCol(self, key=None, selectstep=1):
        """ for getting rid of the useless columns in the dataset
        """
        self.ColumnName = list(self._df.columns)
        for i in self._NonTrainableFeatures:
            if i in self.ColumnName:
                self.ColumnName.remove(i)
        if key is not None:
            self.ColumnName = [i for i in self.ColumnName if key in i]
        self.ColumnName = self.ColumnName[::selectstep]

    def AddPotentialFeatures(self, features):
        """give some strong features you think might be useful.

        Args:
            features: list, the strong features that not in InitialFeatures
        """
        self._PotentialAdd = features

    def SetCCThreshold(self, cc):
        """Set the maximum correlation coefficient between each features

        Args:
            cc: float, the upper bound of correlation coefficient
        """
        self._CoherenceThreshold = cc

    def SetFeaturesLimit(self, FeaturesLimit):
        """Set the features quantity limitation, when selected features reach
           the quantity limitation, the algorithm will exit

        Args:
            FeaturesLimit: int, the features quantity limitation
        """
        self._FeaturesLimit = FeaturesLimit

    def SetTimeLimit(self, TimeLimit):
        """Set the running time limitation, when the running time
           reach the time limit, the algorithm will exit

        Args:
            TimeLimit: double, the maximum time in minutes
        """
        self._TimeLimit = TimeLimit

    def SetSample(self, ratio, samplestate=0, samplemode=1):
        """Set the sample of all data

        Args:
            ratio: double, sample ratio
            samplestate: int, seed
            samplemode: positive int, if 0, every time they
                        sample the same subset, default = 1
        """
        self._sampleratio = ratio
        self._samplestate = samplestate
        self._samplemode = samplemode

    def SetClassifier(self, clf, fit_params=None):
        """Set the classifier and its fit_params

        Args:
            clf: estimator object, defined algorithm to train and evaluate features
            fit_params, dict, optional, parameters to pass to the fit method
        """
        self.clf = clf
        self.fit_params = fit_params = fit_params if fit_params is not None else {}

    def run(self,validate):
        """start running the selecting algorithm

        Args:
            validate: validation method, eq. kfold, last day, etc
        """
        with open(self._logfile, 'a') as f:
            f.write('\n{}\n%{}%\n'.format('Start!','-'*60))
        print("Features Quantity Limit: {}".format(self._FeaturesLimit))
        print("Time Limit: {} min(s)".format(self._TimeLimit))
        a = _LRS_SA_RGSS_combination(df = self._df, clf = self.clf,
                                    RecordFolder = self._logfile,
                                    LossFunction = self._modelscore,
                                    label = self._label,
                                    columnname = self.ColumnName[:],
                                    start = self._temp,
                                    CrossMethod = self._CrossMethod, # your cross term method
                                    PotentialAdd = self._PotentialAdd, # potential feature for Simulated Annealing
                                    Process = [self.Sequence, self.Random, self.Cross],
                                    direction = self._direction, validatefunction = validate,
                                    CoherenceThreshold = self._CoherenceThreshold,
                                    FeaturesQuanLimitation = self._FeaturesLimit,
                                    TimeLimitation = self._TimeLimit,
                                    SampleRatio = self._sampleratio,
                                    SampleState = self._samplestate,
                                    SampleMode = self._samplemode
                                    )
        try:
            a.select()
        finally:
            with open(self._logfile, 'a') as f:
                f.write('\n{}\n{}\n%{}%\n'.format('Done',self._temp,'-'*60))

