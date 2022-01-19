
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
veriler = pd.read_csv("eksikveriler.csv")

#Eksik verileri tamamlama
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(missing_values=np.nan, strategy='mean')

Yas = veriler.iloc[:,1:4].values
print(Yas)

#fit
imputer = imputer.fit(Yas[:,1:4])
#transform
Yas[:,1:4] = imputer.transform(Yas[:,1:4])
print(Yas)


#Kategorik Veri Yapisindan Numerik Veri Yapisina Donusum
ulke = veriler.iloc[:,0:1].values
print(ulke)

from sklearn import preprocessing

le = preprocessing.LabelEncoder()
ulke[:,0] = le.fit_transform(veriler.iloc[:,0])
print(ulke)

ohe = preprocessing.OneHotEncoder()
ulke = ohe.fit_transform(ulke).toarray()
print(ulke)


#numpy dizilerinin dataframe donusumu
sonuc = pd.DataFrame(data=ulke, index=range(22), columns=["fr", "tr", "us"])
print(sonuc)

sonuc2 = pd.DataFrame(data=Yas, index=range(22), columns=["boy", "kilo", "yas"])
print(sonuc2)

cinsiyet = veriler.iloc[:,-1].values
print(cinsiyet)

sonuc3 = pd.DataFrame(data=cinsiyet, index=range(22), columns=["cinsiyet"])
print(sonuc3)

#dataframe'leri birlestime
s = pd.concat([sonuc,sonuc2], axis=1)
print(s)

s2 = pd.concat([s, sonuc3], axis=1)
print(s2)


#Veri Kumesini Egitim Ve Test olarak bolme
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(s,sonuc3,test_size=0.33,random_state=0)

#Oznitelik Olcekleme

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train = sc.fit_transform(x_train)
X_test = sc.fit_transform(x_test)









































print(veriler)
