
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 14 22:18:51 2022

@author: pcc
"""

#kutuphaneler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



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











































print(veriler)
