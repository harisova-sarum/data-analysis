# -*- coding: utf-8 -*-
"""
Created on Wed May 29 13:55:31 2024

@author: Sarum
"""

from sklearn import tree
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from graphviz import Source
import numpy as np
from IPython.display import SVG


data1 = pd.read_csv('dogs.csv')
print(data1.head())
print(data1.info())

data2 = pd.read_csv('cats.csv')
print(data2.head())
print(data2.info())

# Фичи 
X = data2[['Лазает по деревьям']]
# Целевая переменная
y = data2['Вид']

clf = tree.DecisionTreeClassifier(criterion='entropy')
clf.fit(X,y)
Source(tree.export_graphviz(clf, out_file = None,
                             feature_names = list(X),
                             class_names = ['Negative', 'Positive'],
                             filled = True))