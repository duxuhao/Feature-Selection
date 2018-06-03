#!/usr/bin/env python
#-*- coding:utf-8 -*-

##############################################
# File Name: tools.py
# Author: Xuhao Du
# Email: duxuhao88@gmail.com
##############################################

def readlog(fn,score):
    """read the log and export the selected features combination as lost

    Args:
        fn: str, filename
        score: int, score of selected feature combination
    return:
        list: feature combination

    """
    with open(fn,'r') as f:
        a = f.readlines()
        filelength = len(a)

    with open(fn,'r') as f:
        n = 0
        a = f.readline().strip()
        if len(a) < 2:
            a = a + ' '
        while ((a[0] != '*') | (a[1:] != str(score))) & (n < filelength + 1):
            n += 1
            a = f.readline().strip()
            if len(a) < 2:
                a = a + ' '
        if n >= filelength + 1:
            print('[Error]\nlog file has no this score! Please select a correct one\n')
            return None
        a = f.readline().strip().split(' ')
    return a

def filldf(df,features,CrossMethod):
    """fill the dataframe with cross term, cross term must be the same as you construct

    Args:
        df: pandas dataframe
        features: list, features set
        CrossMethod: dict, your cross term method
                    ie {'name':function handel}
    return:
        new df: pandas dataframe, return the original dataframe with cross term 

    """
    for i in CrossMethod.keys():
        for j in features:
            if i in j:
                p = j[1:-1].split(i)
                df[j] = CrossMethod[i](df[p[0]],df[p[1]])
    return df
