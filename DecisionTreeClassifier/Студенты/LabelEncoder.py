# -*- coding: utf-8 -*-
"""
Created on Thu May 30 09:22:15 2024

@author: Sarum
"""



import pandas as pd
import seaborn as sns

from sklearn import tree
from IPython.display import display

df=pd.read_csv('students.csv', delimiter=',')
df.info()

from sklearn import preprocessing
coder = preprocessing.LabelEncoder()
coder.fit(df['Sex'])
coder.transform(df['Sex'])

print(df['Sex'])

df['Sex']=coder.transform(df['Sex'])

for name in ['Coin','Animal','Army']:
    coder.fit(df[name])
    df[name]=coder.transform(df[name])
    
df_cut = df[["Growth","Weight","Sex","Hair length","Children number","Coin","Animal","Army"]]
df_cut = df_cut.dropna()

model = tree.DecisionTreeClassifier(max_depth=4)
model.fit(df_cut[["Growth","Weight","Hair length","Children number","Coin","Animal","Army"]].values.reshape(-1,7),y=df_cut['Sex'].values)

import graphviz
dot_data = tree.export_graphviz(model, out_file=None,
                                feature_names=["Growth","Weight","Hair length","Children number","Coin","Animal","Army"],
                                class_names=['f','m'],
                                filled = True, rounded = True,
                                special_characters = True)
graph = graphviz.Source(dot_data)

#%%
df_test=pd.read_csv('students_test.csv', delimiter=',')
df_test_cut=df_test[["Growth","Weight","Sex","Hair length","Children number","Coin","Animal","Army"]]
df_test_cut=df_test_cut.dropna()

for name in ['Sex','Coin','Animal','Army']:
    coder.fit(df_test_cut[name])
    df_test_cut[name]=coder.transform(df_test_cut[name])

df_test_cut['Predicted'] = model.predict(df_test_cut[['Growth','Weight','Hair length','Children number','Coin','Animal','Army']].values.reshape(-1,7))

pd.crosstab(df_test_cut['Predicted'],df_test_cut['Sex'])

from sklearn.metrics import precision_recall_fscore_support
precision_recall_fscore_support(df_test_cut['Sex'], df_test_cut['Predicted'])