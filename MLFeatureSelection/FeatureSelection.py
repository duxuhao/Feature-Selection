#!/usr/bin/env python
# -*- coding:utf-8 -*-

##############################################
# File Name: FeatureSelection
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


def default_validation(X, y, features, clf, lossfunction, fit_params=None):
    totaltest = []
    kf = KFold(5)
    for train_index, test_index in kf.split(X):
        X_train, X_test = X.ix[train_index, :][features], X.ix[test_index, :][features]
        y_train, y_test = y.ix[train_index, :].Label, y.ix[test_index, :].Label
        clf.fit(X_train, y_train, **fit_params)
        totaltest.append(lossfunction(y_test, clf.predict_proba(X_test)[:, 1]))
    return np.mean(totaltest)


def _reach_limit(func):
    def wrapper(c):
        temp = func(c)
        if (len(c._temp_used_features) >= c._features_limit) | (
                (time.time() - c._start_time) >= c._time_limit):
            print('{0}\nbest score:{1}\nbest {2} features combination: {3}'.format('*-*' * 50,
                                                                                   c._score,
                                                                                   c._features_limit,
                                                                                   c._temp_used_features))
            sys.exit()
        return temp

    return wrapper


class _LRS_SA_RGSS_combination(object):

    def __init__(self, clf, df, record_folder, column_name, start, label, process, direction, loss_function,
                 feature_limit, time_limit, sample_ratio=1, sample_mode=1, sample_state=0, fit_params=None,
                 validate_function=0, potential_add=[], cross_method=0, coherence_threshold=1):
        self._clf = clf
        self._fit_params = fit_params
        self._loss_function = loss_function
        self._df = df
        self._record_folder = record_folder
        self._column_name = column_name
        self._temp_used_features, self._Label = start, label
        self._potential_add = potential_add  # you need to add some potential feature here, otherwise the Simulate Anneal Arithmetic will not work
        self._start_col = ['None']
        self._cross_method = cross_method
        self.process = process
        self._direction = direction
        self._validate_function = validate_function
        if self._validate_function == 0:
            self._validate_function = default_validation  # DefaultValidation is 5-fold
        self._coherence_threshold = coherence_threshold
        self._time_limit = time_limit * 60
        self._features_limit = feature_limit
        self._best_feature = self._temp_used_features[:]
        self._start_time = time.time()
        self._fit_params = fit_params
        if sample_ratio > 1:
            self._sample_ratio = 1
        elif sample_ratio <= 0:
            print("sample ratio should be positive, the set up sample ratio is wrong")
            sys.exit()
        else:
            self._sample_ratio = sample_ratio
        self._sample_state = sample_state
        self._sample_mode = sample_mode

    def _evaluate(self, a, b):
        if self._direction == 'ascend':
            return a > b
        else:
            return a < b

    def select(self):
        self._start_time = time.time()
        # change them based on your evaluation function,
        # if smaller the better, self._score, self._greedyscore = 1, 0
        # if larger the better, self._score, self._greedyscore = 0, 1
        if self._direction == 'ascend':
            self._score, self._greedyscore = 0, 1
        else:
            self._score, self._greedyscore = 1, 0
        self.remain = ''  # for initial
        self._first = 1
        while self._evaluate(self._score, self._greedyscore) | self._first:
            # if the random selection have a better combination,
            # the greedy will loop again. otherwise, the selection complete
            print('test performance of initial features combination')
            self.bestscore, self._best_feature = self._score, self._temp_used_features[:]
            if (self._temp_used_features[:] != []) & (self._first == 1):
                self._validation(self._temp_used_features[:],
                                 str(0), 'baseline', coetest=0)
            # greedy: forward + backward + Simulated Annealing
            if self.process[0]:
                self._greedy()
            self._score_update()
            self._greedyscore = self.bestscore
            print('random select starts with:\n {0}\n score: {1}'.format(self._best_feature,
                                                                         self._greedyscore))
            # random selection
            if self.process[1]:
                self._my_random()

            if self.process[2] & (self._first == 1):  # avoid cross term twice until it is fix
                if 1:  # self._greedyscore == self._score:
                    print('small cycle cross')
                    n = 1
                    while self._score_update() | n:
                        # only if the small cycle cross can construct better features,
                        # then start next small round, otherwise, go to medium cycle
                        self._cross_term_search(self._best_feature, self._best_feature)
                        n = 0
                if self._greedyscore == self._score:
                    print('medium cycle cross')
                    n = 1
                    while self._score_update() | n:
                        # only if the medium cycle cross can construct better features,
                        # then start next medium round, otherwise, go to large cycle
                        self._cross_term_search(self._column_name, self._best_feature)
                        n = 0
                if self._greedyscore == self._score:
                    print('large cycle cross')
                    n = 1
                    while self._score_update() | n:
                        # only if the medium cycle cross can construct better features,
                        # then start next medium round, otherwise, go to large cycle
                        self._cross_term_search(self._column_name, self._column_name)
                        n = 0
            self._first = 0
            self._score_update()
        print('{0}\nbest score:{1}\nbest features combination: {2}'.format('*-*' * 50,
                                                                           self.bestscore,
                                                                           self._best_feature))

    def _validation(self, selectcol, num, addfeature, coetest=0):  # get the score with the new features list and update the best features combination
        """ set up your cross validation here"""
        self.check_limit()
        selectcol = list(OrderedDict.fromkeys(selectcol))
        self._sample_state += self._sample_mode
        if self._sample_ratio < 1:
            tempdf = self._df.sample(frac=self._sample_ratio, random_state=self._sample_state).reset_index(drop=True)
        else:
            tempdf = self._df
        # X, y = self._df, self._df[self._Label]
        X, y = tempdf, tempdf[self._Label]
        totaltest = self._validate_function(X, y, selectcol,
                                            self._clf, self._loss_function)  # , self._fit_params)
        print('Mean loss: {}'.format(totaltest))
        # only when the score improve, the program will record,
        # change the operator ( < or > ) according to your evalulation function
        if self._evaluate(np.mean(totaltest), self._score):
            cc = [0]
            if self._coherence_threshold != 1:  # execute in the features adding process
                coltemp = selectcol[:]
                coltemp.remove(addfeature)
                cc = [pearsonr(self._df[addfeature], self._df[ct])[0] for ct in
                      coltemp]  # check the correlation coefficient
            # to see the correlation coefficient between each two features,
            # not select the feature if its correlation coefficient is too high
            if (np.abs(np.max(cc)) < self._coherence_threshold):
                with open(self._record_folder, 'a') as f:  # record all the imporved combination
                    f.write('{0}  {1}  {2}:\n{3}\t{4}\n'.format(num, addfeature,
                                                                np.abs(np.max(cc)),
                                                                np.round(np.mean(totaltest), 6),
                                                                selectcol[:], '*-' * 50))
                self._temp_used_features, self._score = selectcol[:], np.mean(totaltest)
                if num == 'reverse':
                    self.dele = addfeature
                else:
                    self.remain = addfeature  # updaet the performance

    def _greedy(self):
        col = self._column_name[:]
        print('{0}{1}{2}'.format('-' * 20, 'start greedy', '-' * 20))
        for i in self._temp_used_features:
            print(i)
            try:
                col.remove(i)
            except:
                pass
        self.dele = ''
        self.bestscore, self._best_feature = self._score, self._temp_used_features[:]
        while (self._start_col != self._temp_used_features) | (
                self._potential_add != []):  # stop when no improve for the last round and no potential add feature
            if self._start_col == self._temp_used_features:
                self._score_update()
                if self._direction == 'ascend':
                    self._score *= 0.95  # Simulate Anneal Arithmetic, step back a bit, the value need to be change
                else:
                    self._score /= 0.95
                self._temp_used_features.append(self._potential_add[0])
            print('{0} {1} round {2}'.format('*' * 20, len(self._temp_used_features) + 1, '*' * 20))
            if self.remain in col:
                col.remove(self.remain)
            if self.dele != '':
                col.append(self.dele)
            self._start_col = self._temp_used_features[:]
            for sub, i in enumerate(col):  # forward sequence selection add one each round
                print(i)
                print('{}/{}'.format(sub, len(col)))
                selectcol = self._start_col[:]
                selectcol.append(i)
                self._validation(selectcol, str(1 + sub), i, coetest=0)
            for sr, i in enumerate(self._temp_used_features[
                                   :-1]):  # backward sequence selection, -2 becuase the last 2 is just selected
                deletecol = self._temp_used_features[:]  # can delete several each round
                if i in deletecol:
                    deletecol.remove(i)
                print(i)
                print('reverse {}/{}'.format(sr, len(self._temp_used_features[:-1])))
                self._validation(deletecol, 'reverse', i, coetest=0)
            for i in self._temp_used_features:
                if i in self._potential_add:
                    self._potential_add.remove(i)
        print('{0}{1}{2}'.format('-' * 20, 'complete greedy', '-' * 20))

    def _my_random(self):
        self._score_update()
        col = self._column_name[:]
        print('{0}{1}{2}'.format('-' * 20, 'start random', '-' * 20))
        for i in self._best_feature:
            col.remove(i)
        random.seed(a=self._sample_state)
        for t in range(3, 8):  # add 4 to 8 features randomly, choose your own range
            if t < len(col):
                print('add {} features'.format(t))
                for i in range(50):  # run 50 rounds each quantity, choose your own round number
                    selectcol = random.sample(col, t)
                    recordadd = selectcol[:]
                    for add in self._best_feature:
                        selectcol.append(add)
                    self._validation(selectcol, str(i), str(recordadd))
        print('{0}{1}{2}'.format('-' * 20, 'complete random', '-' * 20))

    @_reach_limit
    def check_limit(self):
        return True

    @_reach_limit
    def _score_update(self):
        if self._direction == 'ascend':
            start = 0
        else:
            start = 1
        if self._score == start:
            return True
        elif self._evaluate(self._score, self.bestscore):
            self.bestscore, self._best_feature = self._score, self._temp_used_features[:]
            return True
        return False

    def _cross_term_search(self, col1, col2):
        self._score_update()
        Effective = []
        crosscount = 0
        for c1 in col1:
            for c2 in col2[::-1]:
                for oper in self._cross_method.keys():
                    print('{}/{}'.format(crosscount, len(self._cross_method.keys()) * len(col1) * len(col2[::-1])))
                    crosscount += 1
                    newcolname = "({}{}{})".format(c1, oper, c2)
                    self._df[newcolname] = self._cross_method[oper](self._df[c1], self._df[c2])
                    selectcol = self._best_feature[:]
                    selectcol.append(newcolname)
                    try:
                        self._validation(selectcol, 'cross term', newcolname, coetest=0)
                    except:
                        pass
                    if self._score_update():
                        Effective.append(newcolname)
                    else:
                        self._df.drop(newcolname, axis=1, inplace=True)
        Effective.remove(self.remain)
        #        for rm in Effective:
        #             self._df.drop(rm, axis = 1, inplace=True)
        self._column_name.append(self.remain)


class Selector(object):
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

    def __init__(self, sequence=True, random=True, cross=True):
        self.sequence = sequence
        self.random = random
        self.cross = cross
        self._non_trainable_features = []
        self._temp = []
        self._log_file = 'record.log'
        self._potential_add = []
        self._cross_method = 0
        self._coherence_threshold = 1
        self._feature_limit = np.inf
        self._time_limit = np.inf
        self._sample_ratio = 1
        self._sample_state = 0
        self._sample_mode = 1

    def set_log_file(self, fn):
        """Setup the log file

        Args:
            fn: str, filename
        """
        self._log_file = fn

    def import_df(self, df, label):
        """Import pandas dataframe to the class

        Args:
            df: pandas dataframe include all features and label.
            label: str, label name
        """
        self._df = df
        self._label = label

    def import_cross_method(self, cross_method):
        """Import a dictionary with different cross function

        Args:
            cross_method: dict, dictionary with different cross function
        """
        self._cross_method = cross_method

    def import_loss_function(self, model_score, direction):
        """Import the loss function

        Args:
            model_score: the function to calculate the loss result
                        with two input series
            direction: str, ‘ascent’ or descent, the way you want
                       the score to go
        """
        self._model_score = model_score
        self._direction = direction

    def initial_features(self, features):
        """Initial your starting features combination

        Args:
            features: list, the starting features combination
        """
        self._temp = features

    def initial_non_trainable_features(self, features):
        """Setting the nontrainable features, eq. user_id

        Args:
            features: list, the nontrainable features
        """
        self._non_trainable_features = features

    def generate_col(self, key=None, selectstep=1):
        """ for getting rid of the useless columns in the dataset
        """
        self.col_names = list(self._df.columns)
        for i in self._non_trainable_features:
            if i in self.col_names:
                self.col_names.remove(i)
        if key is not None:
            self.col_names = [i for i in self.col_names if key in i]
        self.col_names = self.col_names[::selectstep]

    def add_potential_features(self, features):
        """give some strong features you think might be useful.

        Args:
            features: list, the strong features that not in InitialFeatures
        """
        self._potential_add = features

    def set_cc_threshold(self, cc):
        """Set the maximum correlation coefficient between each features

        Args:
            cc: float, the upper bound of correlation coefficient
        """
        self._coherence_threshold = cc

    def set_features_limit(self, features_limit):
        """Set the features quantity limitation, when selected features reach
           the quantity limitation, the algorithm will exit

        Args:
            features_limit: int, the features quantity limitation
        """
        self._feature_limit = features_limit

    def set_time_limit(self, time_limit):
        """Set the running time limitation, when the running time
           reach the time limit, the algorithm will exit

        Args:
            time_limit: double, the maximum time in minutes
        """
        self._time_limit = time_limit

    def set_sample(self, ratio, sample_state=0, sample_mode=1):
        """Set the sample of all data

        Args:
            ratio: double, sample ratio
            sample_state: int, seed
            sample_mode: positive int, if 0, every time they
                        sample the same subset, default = 1
        """
        self._sample_ratio = ratio
        self._sample_state = sample_state
        self._sample_mode = sample_mode

    def set_classifier(self, clf, fit_params=None):
        """Set the classifier and its fit_params

        Args:
            clf: estimator object, defined algorithm to train and evaluate features
            fit_params, dict, optional, parameters to pass to the fit method
        """
        self.clf = clf
        self.fit_params = fit_params = fit_params if fit_params is not None else {}

    def run(self, validate):
        """start running the selecting algorithm

        Args:
            validate: validation method, eq. kfold, last day, etc
        """
        with open(self._log_file, 'a') as f:
            f.write('\n{}\n%{}%\n'.format('Start!', '-' * 60))
        print("Features Quantity Limit: {}".format(self._feature_limit))
        print("Time Limit: {} min(s)".format(self._time_limit))
        a = _LRS_SA_RGSS_combination(clf=self.clf, df=self._df, record_folder=self._log_file,
                                     column_name=self.col_names[:], start=self._temp, label=self._label,
                                     process=[self.sequence, self.random, self.cross], direction=self._direction,
                                     loss_function=self._model_score, feature_limit=self._feature_limit,
                                     time_limit=self._time_limit, sample_ratio=self._sample_ratio,
                                     sample_mode=self._sample_mode, sample_state=self._sample_state,
                                     validate_function=validate, potential_add=self._potential_add,
                                     cross_method=self._cross_method, coherence_threshold=self._coherence_threshold)
        try:
            a.select()
        finally:
            with open(self._log_file, 'a') as f:
                f.write('\n{}\n{}\n%{}%\n'.format('Done', self._temp, '-' * 60))
