# -*- coding: utf-8 -*-
"""
Created on Sun Dec 17 20:38:46 2023

@author: Mehmet
"""

import pandas as pd
import matplotlib.pyplot as plt

# import data
df = pd.read_csv('linear_regression_dataset.csv', sep=';')

x = df.deneyim.values.reshape(-1, 1)
y = df.maas.values.reshape(-1, 1)

# %%
from sklearn.linear_model import LinearRegression

linear_reg = LinearRegression()

linear_reg.fit(x, y)

y_head = linear_reg.predict(x)

# %%
from sklearn.metrics import r2_score

print("r2: ", r2_score(y, y_head))