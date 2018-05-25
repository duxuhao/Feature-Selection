#!/usr/bin/env python
#-*- coding:utf-8 -*-

##############################################
# File Name: coherence_selection.py
# Author: Xuhao Du
# Email: duxuhao88@gmail.com
##############################################

from scipy.stats import pearsonr
from collections import OrderedDict
import random
import time
import numpy as np
import sys

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

class _coherence_selection(object):

    def __init__(self, clf, df, RecordFolder, start, label,
                 direction, LossFunction, FeaturesQuanLimitation, TimeLimitation,
                 fit_params=None, validatefunction=0,coherencelowerbound=0.8,
                 selectbatch=1, selectfrac=1):
        self._clf = clf
        self._fit_params = fit_params
        self._LossFunction = LossFunction
        self._df = df
        self._RecordFolder = RecordFolder
        self._TemplUsedFeatures, self._Label = start, label
        self._Startcol = ['None']
        self._validatefunction = validatefunction
        self._TimeLimitation = TimeLimitation * 60
        self._FeaturesQuanLimitation = FeaturesQuanLimitation
        self._StartTime = time.time()
        self._fit_params = fit_params
        self._frac = selectfrac
        self._batch = selectbatch
        self._lowerbound = coherencelowerbound
        self._direction = direction

    def _evaluate(self, a, b):
        if self._direction == 'ascend':
            return a > b
        else:
            return a < b

    def select(self):
        self._StartTime = time.time()
        if self._direction == 'ascend':
            self._score, self._greedyscore = -np.inf, np.inf
        else:
            self._score, self._greedyscore = np.inf, -np.inf
        print('test performance of initial features combination')
        self.bestscore, self._bestfeature = self._score, self._TemplUsedFeatures[:]
        self._validation(self._TemplUsedFeatures[:], str(0), 'baseline')
        selectcol = self._TemplUsedFeatures[:]
        removelist = selectcol[:]
        if self._frac != 1:
            n = int(len(removelist) * self._frac)
        else:
            n = int(self._batch)
        if n < 1:
            n = 1
        print('Remove Batch: {}'.format(n))
        iter_num = 0
        cc = self._removediag(self._df[selectcol[:]].corr()).abs().max().max()
        print('totally {} features above {}'.format(np.sum(self._removediag(self._df[selectcol[:]].corr()).abs().max() > self._lowerbound),self._lowerbound))
        while (cc >= self._lowerbound) & (len(selectcol) > 1):
            temp = selectcol[:]
            deletenum = 0
            removed = []
            t = self._removediag(self._df[temp].corr())
            cc = t.abs().max().max()
            while (deletenum < n) & (cc >= self._lowerbound):
                tempdelete = t[t.abs().max() == t.abs().max().max()].abs().sum(axis = 1).argmax()
                print('Delete {} with coherence {}'.format(tempdelete,cc))
                temp.remove(tempdelete)
                removed.append(tempdelete)
                deletenum += 1
                t = self._removediag(self._df[temp].corr())
                cc = t.abs().max().max()
            self._validation(temp[:], str(iter_num), str(removed))
            iter_num += 1
            selectcol = temp[:]

        print('{0}\nbest score:{1}\nbest features combination: {2}'.format('*-*' * 50,
                                                                           self.bestscore,
                                                                           self._bestfeature))
        with open(self._RecordFolder, 'a') as f:
            f.write('{0}\nbest score:{1}\nbest features combination: {2}'.format('*-*' * 50,
                                                                           self.bestscore,
                                                                           self._bestfeature))
        return self._bestfeature

    def _validation(self,
                    selectcol,
                    num,
                    rmfeature): #get the score with the new features list and update the best features combination
        """ set up your cross validation here"""
        self.chekcLimit()
        selectcol = list(OrderedDict.fromkeys(selectcol))
        tempdf = self._df
        X, y = tempdf, tempdf[self._Label]
        totaltest = self._validatefunction(X, y, selectcol, self._clf, self._LossFunction) #, self._fit_params)
        print('remove features: {}'.format(rmfeature))
        print('Mean loss: {}'.format(totaltest))
        if self._ScoreUpdate():
            with open(self._RecordFolder, 'a') as f: #record all the imporved combination
                f.write('{0}  {1}:\n{2}\t{3}\n'.format(num, rmfeature,
                                                            np.round(np.mean(totaltest),6),
                                                            selectcol[:], '*-' * 50))
            self._TemplUsedFeatures, self._score = selectcol[:], np.mean(totaltest)

    def _removediag(self,df):
        for i in df.columns:
            df.ix[i,i] = 0
        return df

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


class Select(object):
    """This is a class for importances features selection
    The functions needed to be called before running include:
        ImportDF(pd.dataframe, str) - import you complete dataset including the label column
        ImportLossFunction(func, str) - import your self define loss function,
                                        eq. logloss, accuracy, etc
        InitialFeatures(list) - Initial your starting features combination,
                                if the initial features combination include
                                all features, the backward sequence searching
                                will run automatically
        InitialNonTrainableFeatures(list) - Initial the non-trainable features
        run(func) - start selecting features
    """

    def __init__(self):
        self._NonTrainableFeatures = []
        self._temp = []
        self._logfile = 'record.log'
        self._FeaturesLimit = np.inf
        self._TimeLimit = np.inf
        self._sampleratio = 1
        self._samplestate = 0
        self._samplemode = 1
        self._frac = 1
        self._batch = 1
        self._key = ''
        self._lowerbound = 0.8

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

    def SelectRemoveMode(self, lowerbound = 0.8, frac = 1, batch = 1):
        self._lowerbound = lowerbound
        self._frac = frac
        self._batch = batch

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
        a = _coherence_selection(df = self._df, clf = self.clf,
                                    RecordFolder = self._logfile,
                                    LossFunction = self._modelscore,
                                    label = self._label,
                                    start = self._temp,
                                    FeaturesQuanLimitation = self._FeaturesLimit,
                                    TimeLimitation = self._TimeLimit,
                                    selectbatch = self._batch,
                                    selectfrac = self._frac,
                                    coherencelowerbound = self._lowerbound,
                                    direction = self._direction,
                                    validatefunction = validate,
                                    )
        try:
            best_features_comb = a.select()
        except:
            best_features_comb = a._bestfeature
        finally:
            with open(self._logfile, 'a') as f:
                f.write('\n{}\n{}\n%{}%\n'.format('Done',self._temp,'-'*60))
        return best_features_comb
