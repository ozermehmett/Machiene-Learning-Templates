# -*- coding: utf-8 -*-
"""
Created on Sun Dec 17 18:47:06 2023

@author: Mehmet
"""

import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('polynomial_regression/polynomial_regression.csv', sep=';')

y = df.araba_max_hiz.values.reshape(-1, 1)
x = df.araba_fiyat.values.reshape(-1, 1)

plt.scatter(x, y)
plt.ylabel('araba_max_hiz')
plt.xlabel('araba_fiyat')

# linear regression, y = b0 + b1*x
# multiple regression, y = b0 + b1*x1 + b2*x2

# %% linear regression
from sklearn.linear_model import LinearRegression

lr = LinearRegression()

lr.fit(x, y)

# %% predict
y_head = lr.predict(x)

plt.plot(x, y_head, color='red')
plt.show()

lr.predict([[10000]])


# %% polynomial regression y = b0 + b1*x + b2*x^2 + .... + bn*x^n
from sklearn.preprocessing import PolynomialFeatures

# polynomial_regression = PolynomialFeatures(degree=2)
polynomial_regression = PolynomialFeatures(degree=4)

x_polynomial = polynomial_regression.fit_transform(x)

# %% fit
lr2 = LinearRegression()
lr2.fit(x_polynomial, y)

# %%
y_head2 = lr2.predict(x_polynomial)

plt.plot(x, y_head2, color='red', label='poly')
