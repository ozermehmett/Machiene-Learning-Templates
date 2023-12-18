# -*- coding: utf-8 -*-
"""
Created on Sun Dec 17 20:10:31 2023

@author: Mehmet
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('random_forest_regression/random_forest_regression_dataset.csv', sep=';', header=None)

x = df.iloc[:, 0].values.reshape(-1, 1)
y = df.iloc[:, 1].values.reshape(-1, 1)


# %%
from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(n_estimators=100 , random_state=42)

rf.fit(x, y)

print("7.8 seviyesinde fiyat ne kadar: ", rf.predict([[7.8]]))


x_ = np.arange(min(x), max(x), 0.01).reshape(-1, 1)

y_head = rf.predict(x_)

plt.scatter(x, y, color='red')
plt.plot(x_, y_head, color='green')
plt.xlabel('tribun_level')
plt.ylabel('ucret')
plt.show

