#!/usr/bin/env python
#-*- coding:utf-8 -*-

##############################################
# File Name: tools.py
# Author: Xuhao Du
# Email: duxuhao88@gmail.com
##############################################

def find(s, ch):
    return [i for i in range(len(s)) if s[i:i+len(ch)] == ch]

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

def filldf2(df,features,CrossMethod):
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

    for fea in features:
        if ')' in fea:
            f, tt= [], []
            n = 0
            while n < len(fea):
                if fea[n] in ['(',')']:
                    if len(tt):
                        f.append("df['" + ''.join(tt) + "']")
                    f.append(fea[n])
                    tt = []
                    n += 1
                else:
                    single = True
                    for k in CrossMethod.keys():
                        if fea[n:n+len(k)] == k:
                            if len(tt):
                                f.append("df['" + ''.join(tt) + "']")
                            f.append(k)
                            n += len(k)
                            single = False
                            tt = []
                            break
                    if single:
                        tt.append(fea[n])
                        n += 1
            f = ''.join(f)

            locate_oper = {} #get operator location
            for i in CrossMethod.keys():
                x = find(f,i)
                for j in x:
                    locate_oper[j] = i

            left_bracket = find(f,'(')
            right_bracket = find(f,')')

            rec = [] #get pair braket
            while len(left_bracket) > 0:
                for j in right_bracket:
                    if (j > left_bracket[-1]):
                        rec.append([left_bracket.pop(),j])
                        right_bracket.remove(j)
                        break

            t = [] #pair the operator and braket
            for r in rec:
                for k in locate_oper.keys():
                    if r[0] < k < r[1]:
                        t.append([r[0], locate_oper[k]])
                        del locate_oper[k]
                        break

            CrossMethod_str = {}
            for k in CrossMethod.keys():
                CrossMethod_str[k] = "CrossMethod['{}']".format(k)
            for tt in t:
                f = f[:tt[0]] +CrossMethod_str[tt[1]] + f[tt[0]:]

            for k in CrossMethod.keys():
                n = 0
                while n < len(f)-len(k)-1:
                    if (f[n:n+len(k)] == k) & (f[n+len(k)] != "'"):
                        f = f[:n] + ',' + f[n+len(k):]
                        n -= len(k) + 1
                    n += 1
            print(f)
            df[fea] = eval(f)
    return df
