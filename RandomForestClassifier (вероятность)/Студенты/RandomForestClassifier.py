# -*- coding: utf-8 -*-
"""
Created on Thu May 30 09:22:15 2024

@author: Sarum
"""
#%%
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier

df=pd.read_csv('students.csv', delimiter=',')
df_cut=df[["Growth","Weight","Sex","Hair length","Children number"]]
df_cut=df_cut.dropna()

sns.pairplot(data=df_cut, hue="Sex")

#%%
model = RandomForestClassifier(max_depth=2, random_state=0)
model.fit(df_cut[["Growth","Weight","Hair length","Children number"]].values.reshape(-1,4),y=df_cut['Sex'].values)


df_test=pd.read_csv('students_test.csv', delimiter=',')
df_test_cut=df_test[["Growth","Weight","Sex","Hair length","Children number"]]
df_test_cut=df_test_cut.dropna()

result=model.predict_proba(df_test_cut[["Growth","Weight","Hair length","Children number"]].values.reshape(-1,4))

#результаты вероятности принадлежности к классам
print(result)

#%%
df_test_cut['pr class 0']=result[:,0]
df_test_cut['pr class 1']=result[:,1]
df_test_cut.head()

df_test_cut[(df_test_cut['pr class 1']<0.5)&(df_test_cut['Sex']=='мужской')].head()

df_test_cut[(df_test_cut['pr class 1']<0.9)&(df_test_cut['Sex']=='мужской')].head()

df_test_cut.sort_values(by='pr class 1', ascending=True).head(50)