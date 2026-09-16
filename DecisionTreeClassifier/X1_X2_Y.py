# -*- coding: utf-8 -*-
"""
Created on Wed May 29 13:49:51 2024

@author: Sarum
"""

from sklearn import tree
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from graphviz import Source
import numpy as np
from IPython.display import SVG

data = pd.DataFrame({'X_1': [1, 1, 1, 0, 0, 0, 0, 1],
                     'X_2': [0, 0, 0, 1, 0, 0, 0, 1],
                     'Y': [1, 1, 1, 1, 0, 0, 0, 0]})

print(data)

clf = tree.DecisionTreeClassifier(criterion='entropy')

# Фичи 
X = data[['X_1','X_2']]
# Целевая переменная
y = data.Y

clf.fit(X,y)

graph = Source(tree.export_graphviz(clf, out_file = None,
                             feature_names = list(X),
                             class_names = ['Negative', 'Positive'],
                             filled = True))