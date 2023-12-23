# -*- coding: utf-8 -*-
"""
Created on Sat Dec 23 22:21:34 2023

@author: Mehmet
"""

import pandas as pd

# %% import X(twitter) data
data = pd.read_csv('Machine-Learning-Templates/NLP/gender_classifier.csv', encoding='latin1')
data = pd.concat([data.gender, data.description], axis=1)
data.dropna(axis=0, inplace=True)

data.gender = [1 if each == 'female' else 0 for each in data.gender]

# %%
from nltk.corpus import stopwords
import nltk as nlp
import re

# %%
description_list = list()

for description in data.description:
    # cleaning data
    description = re.sub('[^a-zA-Z]', ' ', description)
    description = description.lower()
    
    # tokenize
    description = nlp.word_tokenize(description)
    
    # stopwords
    # description = [word for word in description if not word in set(stopwords.words('english'))]
    
    # Lemmatization loved --> love
    lemma = nlp.WordNetLemmatizer()
    description = [lemma.lemmatize(word) for word in description]
    
    description = ' '.join(description)
    
    description_list.append(description)
    
# %% bag of words
from sklearn.feature_extraction.text import CountVectorizer
max_features = 500

count_vectorizer = CountVectorizer(max_features=max_features, stop_words='english')

# x
sparce_matrix = count_vectorizer.fit_transform(description_list).toarray()

# print('en sık kullanılan {} kelimeller {}'.format(max_features, count_vectorizer.get_feature_names_out()))

# %%
y = data.iloc[:, 0].values
x = sparce_matrix

# %% train test split
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=42)

# %% naive bayes
from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()
nb.fit(x_train, y_train)

# %% prediction
y_pred = nb.predict(x_test)

from sklearn.metrics import accuracy_score
print('accuracy:', accuracy_score(y_pred, y_test))

