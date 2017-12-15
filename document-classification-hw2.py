from __future__ import print_function
import nltk
from multiprocessing import Pool
from itertools import chain
import logging
import numpy as np
import sys
from time import time
import matplotlib.pyplot as plt

from sklearn.datasets import load_files
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from sklearn.utils.extmath import density
from sklearn import metrics


# In[3]:

dataset_folder = "./ohsumed-first-20000-docs"


# In[ ]:

training_folder = dataset_folder + '/training'
test_folder = dataset_folder + '/test'


# In[ ]:

test_data = load_files(container_path=test_folder, categories=None, load_content=True, shuffle=True, encoding='utf-8', random_state=17)
training_data = load_files(container_path=training_folder, categories=None, load_content=True, shuffle=True, encoding='utf-8', random_state=17)


# In[ ]:

categories = training_data.target_names


# In[ ]:

print("categories values: ",categories)


# In[ ]:

y_train,y_test = training_data.target , test_data.target


# In[ ]:

training_data.data[0]


# In[ ]:


# nltk.download()
punkt = nltk.data.load('tokenizers/punkt/english.pickle')
print("hi")
catagory_stats = {}
for train_data_file, target in zip(training_data.data, training_data.target):
    sentences = punkt.tokenize(train_data_file.lower())
    p = Pool()
    words = list(chain.from_iterable(p.map(nltk.tokenize.word_tokenize, sentences)))
    p.close()
    # Now remove words that consist of only punctuation characters
    words = [word for word in words if not all(char in string.punctuation for char in word)]
    # Remove contractions - wods that begin with '
    words = [word for word in words if not (word.startswith("'") and len(word) <=2)]
    if target in catagory_stats:
        words_stats[target].append(words)
    else:
        words_stats[target] = words
    print(len(words))


# In[ ]:

from collections import Counter
c = Counter(words)
c.most_common(n=10)


# In[ ]:

stopwords = nltk.corpus.stopwords.words('english')
new_c = c.copy()
for key in c:
    if key in stopwords:
        del new_c[key]
new_c.most_common(n=10)

