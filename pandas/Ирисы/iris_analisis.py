# -*- coding: utf-8 -*-
"""
Created on Wed May 29 10:08:50 2024

@author: Sarum
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

iris = pd.read_csv('iris.data.txt')
attributes = ["sepal_length", "sepal_width", "petal_length", "petal_width", "class"]
iris.columns = attributes

print(iris.head())
df = iris.drop('class', axis = 1)
print(df.head())

plt.figure(figsize=[15,12])
for column in df:
   sns.distplot(df[column])
   plt.legend(labels= ['sepal_length',	'sepal_width',	'petal_length',	'petal_width'])
#%%   
sns.violinplot(y=iris["petal_length"])