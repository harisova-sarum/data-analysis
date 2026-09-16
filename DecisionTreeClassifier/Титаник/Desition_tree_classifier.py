# -*- coding: utf-8 -*-
"""
Created on Wed May 29 14:03:38 2024

@author: Sarum
"""

from sklearn import tree
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from graphviz import Source
import numpy as np
from IPython.display import SVG
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

titanic_data = pd.read_csv("titanic.csv")
print(titanic_data.head())

print(titanic_data.isnull().sum())

#%%
X = titanic_data.drop(['PassengerId','Survived','Name','Ticket','Cabin'], axis = 1)
y = titanic_data.Survived

X = pd.get_dummies(X)

X = X.fillna({'Age': X.Age.median()})

X.isnull().sum()

#%%
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.33, random_state = 42)

print(X_train.shape, X_test.shape)

clf = tree.DecisionTreeClassifier(criterion = 'entropy', max_depth = 3)
clf.fit(X_train,y_train)
clf.score(X_train,y_train)
clf.score(X_test,y_test)

#%%
max_depth_values = range(1,100)
scors_data = pd.DataFrame()

for max_depth in max_depth_values:
  clf = tree.DecisionTreeClassifier(criterion = 'entropy', max_depth = max_depth)
  clf.fit(X_train,y_train)
  train_score = clf.score(X_train,y_train)
  test_score = clf.score(X_test,y_test)

  temp_score_data = pd.DataFrame({'max_depth': [max_depth],
                                  'train_score': [train_score],
                                  'test_score': [test_score]})
  scors_data = pd.concat([scors_data, temp_score_data], ignore_index=True)
  
print(scors_data.head())  
  
scors_data_long = pd.melt(scors_data, id_vars = ['max_depth'], value_vars = ['train_score','test_score'],
                          var_name = 'set_type', value_name = 'score') 

print(scors_data_long.head())

sns.lineplot(x= 'max_depth',y= 'score', hue = 'set_type', data = scors_data_long)

#%% Кросс-валидация
clf = tree.DecisionTreeClassifier(criterion = 'entropy', max_depth = 4)

cross_val_score(clf, X_train, y_train, cv = 5)

max_depth_values = range(1,100)
scors_data = pd.DataFrame()

for max_depth in max_depth_values:
  clf = tree.DecisionTreeClassifier(criterion = 'entropy', max_depth = max_depth)
  clf.fit(X_train,y_train)
  train_score = clf.score(X_train,y_train)
  test_score = clf.score(X_test,y_test)
  mean_cross_val_score = cross_val_score(clf, X_train, y_train, cv = 5).mean()

  temp_score_data = pd.DataFrame({'max_depth': [max_depth],
                                  'train_score': [train_score],
                                  'test_score': [test_score],
                                  'cross_val_score': [mean_cross_val_score]})
  scors_data = pd.concat([scors_data, temp_score_data], ignore_index=True)

  
print(scors_data.head())

scors_data_long = pd.melt(scors_data, id_vars = ['max_depth'], value_vars = ['train_score','test_score','cross_val_score'],
                          var_name = 'set_type', value_name = 'score')

sns.lineplot(x= 'max_depth',y= 'score', hue = 'set_type', data = scors_data_long)

best_clf = tree.DecisionTreeClassifier(criterion = 'entropy', max_depth = 9)

print(cross_val_score(best_clf, X_test, y_test, cv = 5).mean())