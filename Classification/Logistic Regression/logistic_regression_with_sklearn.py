# -*- coding: utf-8 -*-
"""
Created on Tue Dec 19 15:49:04 2023

@author: Mehmet
"""

# %% Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#%% read csv
data = pd.read_csv('Machiene-Learning-Templates/Classification/Logistic Regression/data.csv', sep=',')

data.drop(['Unnamed: 32', 'id'], axis=1, inplace=True)

data.diagnosis = [1 if each == "M" else 0 for each in data.diagnosis]

y = data.diagnosis.values
x_data = data.drop(['diagnosis'], axis=1)

# %% normalization
x = ((x_data - np.min(x_data)) / (np.max(x_data) - np.min(x_data))).values

# (x - min(x)) / (max(x) - min(x))

# %% train test split
 from sklearn.model_selection import train_test_split
 
 x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
 
 x_train = x_train.T
 x_test = x_test.T
 y_train = y_train.T
 y_test = y_test.T
 
 print("x_train:", x_train.shape) 
 print("x_test:", x_test.shape)
 print("y_train:", y_train.shape)
 print("y_test:", y_test.shape)
 
 
 # %% sklearn with lr
 from sklearn.linear_model import LogisticRegression
 
 lr = LogisticRegression()
 
 lr.fit(x_train.T, y_train.T)
 
 print("test accuracy: {} %".format(lr.score(x_test.T, y_test.T)))
 