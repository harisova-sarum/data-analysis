# -*- coding: utf-8 -*-
"""
Created on Thu May 30 09:22:15 2024

@author: Sarum
"""
#%%
import pandas as pd
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier

#%%
df=pd.read_csv('students.csv', delimiter=',')
df_cut =df[["Weight","Growth","Sex"]] 
df_cut.head()

df_cut = df_cut.dropna()

sns.scatterplot(data=df_cut,x="Weight",y="Growth",hue='Sex')

#%% Гипотеза: мужчины и женщины образуют два облака. Причем для многих точек выполнено условие: если соседние точки мужчины (женщины), то данная точка мужчина (женщина).

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
#обучение самого нормировщика
scaler.fit(df_cut[["Weight","Growth"]].values.reshape(-1,2))
#нормирование
arr= scaler.transform(df_cut[["Weight","Growth"]].values.reshape(-1,2))

# n_neighbors=1 - самый ближайший  сосед
model = KNeighborsClassifier(n_neighbors=3)
model.fit(arr, y=df_cut["Sex"].values)

df_test=pd.read_csv('students_test.csv', delimiter=',')
df_test_cut=df_test[["Weight","Growth", "Sex"]]
df_test_cut=df_test_cut.dropna()

arr_test= scaler.transform(df_test_cut[["Weight","Growth"]].values.reshape(-1,2))


df_test_cut['Predicted']=model.predict(arr_test)
pd.crosstab(df_test_cut['Predicted'], df_test_cut['Sex'])

df_test_cut['Code']='0'
df_test_cut.loc[(df_test_cut['Sex']=='мужской')&(df_test_cut['Predicted']=='женский'),'Code']='1'
df_test_cut.loc[(df_test_cut['Sex']=='женский')&(df_test_cut['Predicted']=='мужской'),'Code']='2'

sns.scatterplot(data=df_test_cut,x='Weight',y='Growth', hue='Code')

#%%
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

print(accuracy_score(df_test_cut['Predicted'],df_test_cut['Sex']))
#%%
print(precision_score(df_test_cut['Predicted'],df_test_cut['Sex'],pos_label='женский'))
#%%
print(precision_score(df_test_cut['Predicted'],df_test_cut['Sex'],pos_label='мужской'))
#%%
print(recall_score(df_test_cut['Predicted'],df_test_cut['Sex'],pos_label='женский'))
#%%
print(recall_score(df_test_cut['Predicted'],df_test_cut['Sex'],pos_label='мужской'))