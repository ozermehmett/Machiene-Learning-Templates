# -*- coding: utf-8 -*-
"""
Created on Sat Dec 23 00:19:53 2023

@author: Mehmet
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# %% create dataset

# class1
x1 = np.random.normal(25, 5, 1000)
y1 = np.random.normal(25, 5, 1000)

# class2
x2 = np.random.normal(55, 5, 1000)
y2 = np.random.normal(60, 5, 1000)

# class3
x3 = np.random.normal(55, 5, 1000)
y3 = np.random.normal(15, 5, 1000)

# %%
x = np.concatenate((x1, x2, x3), axis=0)
y = np.concatenate((y1, y2, y3), axis=0)

dictionary = {'x': x, 'y': y}

# %%
data = pd.DataFrame(dictionary)

# %%
from scipy.cluster.hierarchy import linkage, dendrogram

merg = linkage(data, method='ward')
dendrogram(merg, leaf_rotation=90)

plt.xlabel('data points')
plt.ylabel('euclidean distance')
plt.show()
