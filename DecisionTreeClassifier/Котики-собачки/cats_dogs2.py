# -*- coding: utf-8 -*-
"""
Created on Wed May 29 16:47:55 2024

@author: Sarum
"""

from sklearn import tree
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from graphviz import Source
import numpy as np
from IPython.display import SVG
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

cats_and_dogs = pd.read_csv('dogs_n_cats.csv')
print(cats_and_dogs.head())

X = cats_and_dogs.drop(['Вид'], axis = 1)
y = cats_and_dogs['Вид']

le = preprocessing.LabelEncoder()
le.fit(y)
le.transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

max_depth_values = range(1,100)

scors_data = pd.DataFrame()

for max_depth in max_depth_values:
  clf = tree.DecisionTreeClassifier(criterion = 'entropy', max_depth = max_depth)
  clf.fit(X_train,y_train)
  train_score = clf.score(X_train,y_train)
  mean_cross_val_score = cross_val_score(clf, X_train, y_train, cv = 5).mean()

  temp_score_data = pd.DataFrame({'max_depth': [max_depth],
                                  'train_score': [train_score],
                                  'cross_val_score': [mean_cross_val_score]})
  scors_data = pd.concat([scors_data, temp_score_data], ignore_index=True)
  
scors_data_long = pd.melt(scors_data, id_vars = ['max_depth'], value_vars = ['train_score','cross_val_score'],
                          var_name = 'set_type', value_name = 'score') 

print(scors_data_long.head())

sns.lineplot(x= 'max_depth',y= 'score', hue = 'set_type', data = scors_data_long)
  