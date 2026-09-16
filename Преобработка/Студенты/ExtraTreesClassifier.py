# -*- coding: utf-8 -*-
"""
Created on Thu May 30 14:13:55 2024

@author: Sarum
"""

import pandas as pd
import seaborn as sns

from sklearn import tree
from sklearn import preprocessing

df=pd.read_csv('students.csv', delimiter=',')
df=df.dropna()
df.info()

coder = preprocessing.LabelEncoder()
for name in ['Sex','Coin','Animal','Army','Glasses','Your rating in university',
             'Fastfood','Hostel','Chocolate','Brother-sister','Plane seat', 'Problems in last semester',
             'Rock paper scissors', 'Strange people', 'Your insitute']:
    coder.fit(df[name])
    df[name]=coder.transform(df[name])

#%% Важность каждого признака в задаче предсказания пола человека
from sklearn.ensemble import ExtraTreesClassifier
selector = ExtraTreesClassifier()

result = selector.fit(df[df.columns],df['Sex'])
result.feature_importances_

features_table = pd.DataFrame(result.feature_importances_, index=df.columns, columns = ['importance'])

features_table.sort_values(by='importance', ascending = False)

#%%
df=pd.read_csv('students.csv', delimiter=',')
df_cut=df[['Army','Shoe size','Growth','Hair length','Computer science rating','Coin','Weight','Sex']]
df_cut=df_cut.dropna()
df_cut.info()

df_cut= pd.get_dummies(df_cut, drop_first=True)
df_cut.head()

#%% Угадать пол человека можно за небольшое количество вопросов. Модель найдет эти вопросы автоматически
model = tree.DecisionTreeClassifier(max_depth=3)
model.fit(df_cut[['Army_не призовут (по разным причинам)','Shoe size','Hair length','Growth','Coin_Решка','Computer science rating','Weight']].values.reshape(-1,7),y=df_cut['Sex_мужской'])

import graphviz
dot_data = tree.export_graphviz(model, out_file=None,
                                feature_names=['Army_не призовут (по разным причинам)','Shoe size',"Hair length",'Growth','Coin_Решка','Computer science rating','Weight'],
                                class_names=['f','m'],
                                filled = True, rounded = True,
                                special_characters = True)
graph = graphviz.Source(dot_data)


df_test=pd.read_csv('students_test.csv', delimiter=',')
df_test_cut=df_test[["Army","Shoe size","Hair length","Growth","Coin","Computer science rating","Weight","Sex"]]
df_test_cut=df_test_cut.dropna()

df_test_cut= pd.get_dummies(df_test_cut, drop_first=True)
df_test_cut.head()

df_test_cut['Predicted']=model.predict(df_test_cut[['Army_не призовут (по разным причинам)','Shoe size',"Hair length",'Growth','Coin_Решка','Computer science rating','Weight']].values.reshape(-1,7))

pd.crosstab(df_test_cut['Predicted'],df_test_cut['Sex_мужской'])