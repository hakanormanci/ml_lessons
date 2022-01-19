# -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 18:50:13 2020

@author: sadievrenseker
"""

#1.kutuphaneler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# veri yukleme
ana_tablo = pd.read_csv('test_v1.csv')
ana_tablo_v2 = ana_tablo.iloc[:,1:22]

# veri donusum
le = preprocessing.LabelEncoder()
ohe = preprocessing.OneHotEncoder()

teamid = ana_tablo_v2.iloc[:,0:1].values
teamid = ohe.fit_transform(teamid).toarray()
teamid = pd.DataFrame(data=teamid, columns=['alba','efes','monaco','milano','baskonia','crvena','cska','barca','bayern','fener','asvel','maccabi','oly','pana','madrid','unics','zalgiris','zenit'])

home = ana_tablo_v2.iloc[:,1:2].values
home[:,0] = le.fit_transform(ana_tablo_v2.iloc[:,1:2])

win = ana_tablo_v2.iloc[:,-1:].values
win[:,0] = le.fit_transform(ana_tablo_v2.iloc[:,-1:])

# train&test split
x_tablo = pd.concat([teamid,ana_tablo_v2.iloc[:,2:19]],axis=1)
x_tablo.fillna(value=0, inplace=True)
x_train,x_test,y_train,y_test = train_test_split(x_tablo,win,test_size=0.33,random_state=0)

# linear regression
lin_reg = LinearRegression()
lin_reg.fit(x_train,y_train)
y_pred = lin_reg.predict(x_test)




















x = veriler.iloc[:,1:2]
y = veriler.iloc[:,2:]
X = x.values
Y = y.values


#linear regression
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X,Y)

plt.scatter(X,Y,color='red')
plt.plot(x,lin_reg.predict(X), color = 'blue')
plt.show()


#polynomial regression
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 2)
x_poly = poly_reg.fit_transform(X)
print(x_poly)
lin_reg2 = LinearRegression()
lin_reg2.fit(x_poly,y)
plt.scatter(X,Y,color = 'red')
plt.plot(X,lin_reg2.predict(poly_reg.fit_transform(X)), color = 'blue')
plt.show()

from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
x_poly = poly_reg.fit_transform(X)
print(x_poly)
lin_reg2 = LinearRegression()
lin_reg2.fit(x_poly,y)
plt.scatter(X,Y,color = 'red')
plt.plot(X,lin_reg2.predict(poly_reg.fit_transform(X)), color = 'blue')
plt.show()

#tahminler

print(lin_reg.predict([[11]]))
print(lin_reg.predict([[6.6]]))

print(lin_reg2.predict(poly_reg.fit_transform([[6.6]])))
print(lin_reg2.predict(poly_reg.fit_transform([[11]])))










