# -*- coding: utf-8 -*-
"""
Created on Sat Dec 23 20:38:11 2023

@author: Mehmet
"""

import pandas as pd
import numpy as np

# %% import X(twitter) data
data = pd.read_csv('Machine-Learning-Templates/NLP/gender_classifier.csv', encoding='latin1')
data = pd.concat([data.gender, data.description], axis=1)
data.dropna(axis=0, inplace=True)

data.gender = [1 if each == 'female' else 0 for each in data.gender]

# %% cleaning data
# Regular Expression RE
import re

first_description = data.description[4]
description = re.sub('[^a-zA-Z]', ' ', first_description)

description = description.lower()

# %% stopwords(Irrelavant Words)
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
from nltk.corpus import stopwords

# %% tokenizer
# description = description.split() # 'shouldn't' --> 'shouldn't'

# with tokenizer
description = nltk.word_tokenize(description) # shouldn't --> 'should' 'n't'

# %%
description = [word for word in description if not word in set(stopwords.words('english'))]

# %% Lemmatization loved --> love
import nltk as nlp

lemma = nlp.WordNetLemmatizer()

description = [lemma.lemmatize(word) for word in description]

# %%
description_ = ' '.join(description)
















