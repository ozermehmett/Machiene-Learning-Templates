# -*- coding: utf-8 -*-
"""
Created on Sun Dec 24 20:53:51 2023

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


# %% Grid Search Cross Validation With Logistic Regression
x = x[:100, :]
y = y[:100]

from sklearn.linear_model import LogisticRegression

grid = {'C': np.logspace(-3, 3, 7), 'penalty': ['l1', 'l2']} # l1 = lasso, l2 = ridge

logreg = LogisticRegression()

from sklearn.model_selection import GridSearchCV
logreg_cv = GridSearchCV(estimator=logreg, param_grid=grid, cv=10)

logreg_cv.fit(x_train, y_train)

# %% print hyperparameter k value in knn algorithm 
print('tuned hyperparameter:', logreg_cv.best_params_)
print('The best accuracy value according to the tuned parameter:', logreg_cv.best_score_)

# %%
logreg2 = LogisticRegression(C=0.1, penalty='l2')
logreg2.fit(x_train, y_train)

print('test accuracy:', logreg2.score(x_test, y_test))












