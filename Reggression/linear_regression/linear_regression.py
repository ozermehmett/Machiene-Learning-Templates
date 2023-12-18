# -*- coding: utf-8 -*-
"""
Created on Sun Dec 17 13:04:53 2023

@author: Mehmet
"""

# import library
import pandas as pd
import matplotlib.pyplot as plt

# import data
df = pd.read_csv('linear_regression_dataset.csv', sep=';')

# plot data
plt.scatter(df.deneyim, df.maas)
plt.xlabel('deneyim')
plt.ylabel('maas')
plt.show()


#%% Linear Regression

# sklearn library
from sklearn.linear_model import LinearRegression

# linear reg model
linear_reg = LinearRegression()

x = df.deneyim.values.reshape(-1, 1)
y = df.maas.values.reshape(-1, 1)

linear_reg.fit(x, y)

#%% prediction
import numpy as np

b0 = linear_reg.predict([[0]])
print("b0: ", b0)

b0_ = linear_reg.intercept_
print("b0_: ", b0_) # intercept(y eksenini kestiği nokta)

b1 = linear_reg.coef_
print("b1: ", b1) # slope(eğim)

# mass = 1663 + 1138 * deneyim

maas_yeni = 1663 + 1138 * 11
print(maas_yeni)

maas_yeni_11 = linear_reg.predict([[11]])
print(maas_yeni_11)

# visualize line (deneyim)
array = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]).reshape(-1, 1)

plt.scatter(x, y)

y_head = linear_reg.predict(array)

plt.plot(array, y_head, color='red')
