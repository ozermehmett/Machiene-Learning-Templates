# -*- coding: utf-8 -*-
"""
Created on Wed Dec 20 14:49:31 2023

@author: Mehmet
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# %%
data = pd.read_csv('Classification/Support Vector Machine/data.csv')

data.drop(['id', 'Unnamed: 32'], axis=1, inplace=True)
data.tail()

# %%
M = data[data.diagnosis == 'M']
B = data[data.diagnosis == 'B']

# %%
# scatter plot
plt.scatter(M.radius_mean, M.texture_mean, color='red', label='M', alpha=0.4)
plt.scatter(B.radius_mean, B.texture_mean, color='green', label='B', alpha=0.4)
plt.xlabel('texture_mean')
plt.ylabel('radius_mean')
plt.legend()
plt.show()

# %%
data.diagnosis = [1 if each == 'M' else 0 for each in data.diagnosis]
y = data.diagnosis.values
x_data = data.drop(['diagnosis'], axis=1)

# %%
# normaliaztion
x = ((x_data - np.min(x_data)) / (np.max(x_data) - np.min(x_data))).values

# %% 
# train test split
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1)

# %% SVM
from sklearn.svm import SVC
svm = SVC(random_state=1)

svm.fit(x_train, y_train)

# %% Score
print('SVM score accuracy: {} '.format(svm.score(x_test, y_test)))
