# -*- coding: utf-8 -*-
"""
Created on Sun Dec 24 20:14:32 2023

@author: Mehmet
"""

from sklearn.datasets import load_iris
import pandas as pd
import numpy as np

# %%
iris = load_iris()

x = iris.data
y = iris.target

# %% normalization
x = (x - np.min(x)) / (np.max(x) - np.min(x))

# %% train test split
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

# %% knn model
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=3)

# %% K-Fold Cross Validation K = 10
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator=knn, X=x_train, y=y_train, cv=10)

print('average accuracy:', np.mean(accuracies))
print('average std:', np.std(accuracies))

# %%
knn.fit(x_train, y_train)
print('test accuracy:', knn.score(x_test, y_test))

# %% Grid Search Cross Validation
from sklearn.model_selection import GridSearchCV

grid = {'n_neighbors': np.arange(1, 50)}

knn = KNeighborsClassifier()

knn_cv = GridSearchCV(estimator=knn, param_grid=grid, cv=10)
knn_cv.fit(x, y)

# %% print hyperparameter k value in knn algorithm 
print('tuned hyperparameter:', knn_cv.best_params_)
print('The best accuracy value according to the tuned parameter:', knn_cv.best_score_)
