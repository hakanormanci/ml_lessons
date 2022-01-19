
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 14 22:18:51 2022

@author: pcc
"""

#1. kutuphaneler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#2. veri on isleme
#2.1 veri yukleme
veriler = pd.read_csv("satislar.csv")
print(veriler)


aylar = veriler[["Aylar"]]
print(aylar)

satislar = veriler[["Satislar"]]
print(satislar)

satislar2= veriler.iloc[:,1:2].values
print(satislar2)


#Veri Kumesini Egitim Ve Test olarak bolme
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(aylar,satislar,test_size=0.33,random_state=0)

"""
#Oznitelik Olcekleme

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train = sc.fit_transform(x_train)
X_test = sc.fit_transform(x_test)
Y_train = sc.fit_transform(y_train)
Y_test = sc.fit_transform(y_test)

"""
#model insasi (linear regression)
from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lr.fit(x_train,y_train)

tahmin = lr.predict(x_test)

x_train = x_train.sort_index()
y_train = y_train.sort_index()

plt.plot(x_train,y_train)
plt.plot(x_test,tahmin)

plt.title("Aylara Gore Satislar")
plt.xlabel("Aylar")
plt.ylabel("Satislar")










































print(veriler)
