# -*- coding: utf-8 -*-
"""
Created on Thu Dec 21 19:35:03 2023

@author: Mehmet
"""

import pandas as pd
import numpy as np

# %%
data = pd.read_csv('Classification/Random Forest/data.csv')

data.drop(['id', 'Unnamed: 32'], axis=1, inplace=True)
data.tail()

# %%

M = data[data.diagnosis == 'M']
B = data[data.diagnosis == 'B']

# %%
data.diagnosis = [1 if each == 'M' else 0 for each in data.diagnosis]
y = data.diagnosis.values
x_data = data.drop(['diagnosis'], axis=1)

# %%
x = ((x_data - np.min(x_data)) / (np.max(x_data) - np.min(x_data))).values

# %%
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)

# %%
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=100, random_state=42)

rf.fit(x_train, y_train)

#%%
print('Random Forest score accuracy: {} '.format(rf.score(x_test, y_test)))
