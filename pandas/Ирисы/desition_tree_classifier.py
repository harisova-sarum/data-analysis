# -*- coding: utf-8 -*-
"""
Created on Wed May 29 14:47:01 2024

@author: Sarum
"""

from sklearn import tree
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from graphviz import Source
import numpy as np
from IPython.display import SVG

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

iris = pd.read_csv('train_iris.csv')
print(iris.head())

X = iris.drop(['Unnamed: 0','species'], axis = 1)
y = iris.species

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.5, random_state = 42)

max_depth_values = range(1,100)

for max_depth in max_depth_values:
  clf = tree.DecisionTreeClassifier(random_state=0,criterion = 'entropy', max_depth = max_depth)
  clf.fit(X_train,y_train)
  train_score = clf.score(X_train,y_train)
  test_score = accuracy_score(y_train,y_test)

  temp_score_data = pd.DataFrame({'max_depth': [max_depth],
                                  'train_score': [train_score],
                                  'test_score': [test_score]})
  scors_data = scors_data.append(temp_score_data)
  
scors_data_long = pd.melt(scors_data, id_vars = ['max_depth'], value_vars = ['train_score','test_score'],
                          var_name = 'set_type', value_name = 'score')

print(scors_data_long)