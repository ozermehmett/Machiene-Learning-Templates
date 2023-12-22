# -*- coding: utf-8 -*-
"""
Created on Fri Dec 22 23:46:24 2023

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
from sklearn.cluster import KMeans

wcss = list()

for k in range(1, 15):
    k_means = KMeans(n_clusters=k)
    
    k_means.fit(data)
    
    wcss.append(k_means.inertia_)
    
# %% 
plt.plot(range(1, 15), wcss)
plt.xlabel('number of k value')
plt.ylabel('wcss')
plt.show()

# %% k=3 kmeans model
k_means2 = KMeans(n_clusters=3)
clusters = k_means2.fit_predict(data)

data['label'] = clusters

# %%
plt.scatter(data.x[data.label == 0], data.y[data.label == 0], color='red')
plt.scatter(data.x[data.label == 1], data.y[data.label == 1], color='blue')
plt.scatter(data.x[data.label == 2], data.y[data.label == 2], color='green')
plt.scatter(k_means2.cluster_centers_[:, 0], k_means2.cluster_centers_[:, 1], color='yellow')
plt.show()
