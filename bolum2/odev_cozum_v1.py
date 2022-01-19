#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 15 04:18:20 2018

@author: sadievrenseker
"""

#1. kutuphaneler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#2. Veri Onisleme

#2.1. Veri Yukleme
veriler = pd.read_csv('odev_tenis.csv')
#pd.read_csv("veriler.csv")


#data preprocessing

play = veriler.iloc[:,4:].values
outlook = veriler.iloc[:,0:1].values
windy = veriler.iloc[:,3:4].values

from sklearn import preprocessing
le = preprocessing.LabelEncoder()
ohe = preprocessing.OneHotEncoder()

windy[:,0] = le.fit_transform(veriler.iloc[:,3:4])
play[:,0] = le.fit_transform(veriler.iloc[:,4])
outlook[:,0] = le.fit_transform(veriler.iloc[:,0])
outlook = ohe.fit_transform(outlook).toarray()

havadurumu = pd.DataFrame(data = outlook, index=range(14), columns=['o','r','s'])
play2 = pd.DataFrame(data=play, index=range(14), columns=['play'])
windy2 = pd.DataFrame(data=windy, index=range(14), columns=['windy'])
sonveriler = pd.concat([havadurumu,veriler.iloc[:,1:3],windy2,play2], axis=1)

#cross validation & prediction

from sklearn.model_selection import train_test_split
x_train, x_test,y_train,y_test = train_test_split(sonveriler.iloc[:,:-1],sonveriler.iloc[:,-1:],test_size=0.33, random_state=0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)

y_pred = regressor.predict(x_test)

print(y_pred)


#backward elimination

import statsmodels.api as sm

X = np.append(arr=np.ones((14,1)).astype(int), values=sonveriler.iloc[:,:-1], axis=1)
X_l = sonveriler.iloc[:,[0,1,2,3,4,5]].values

r_ols = sm.OLS(endog=sonveriler.iloc[:,-1:], exog=X_l)
r = r_ols.fit()
print(r.summary())
"""

import statsmodels.formula.api as sm 
X = np.append(arr = np.ones((14,1)).astype(int), values=sonveriler.iloc[:,:-1], axis=1 )
X_l = sonveriler.iloc[:,[0,1,2,3,4,5]].values
r_ols = sm.OLS(endog = sonveriler.iloc[:,-1:], exog =X_l)
r = r_ols.fit()
print(r.summary())

sonveriler = sonveriler.iloc[:,1:]

import statsmodels.formula.api as sm 
X = np.append(arr = np.ones((14,1)).astype(int), values=sonveriler.iloc[:,:-1], axis=1 )
X_l = sonveriler.iloc[:,[0,1,2,3,4]].values
r_ols = sm.OLS(endog = sonveriler.iloc[:,-1:], exog =X_l)
r = r_ols.fit()
print(r.summary())

x_train = x_train.iloc[:,1:]
x_test = x_test.iloc[:,1:]

regressor.fit(x_train,y_train)


y_pred = regressor.predict(x_test)
"""






