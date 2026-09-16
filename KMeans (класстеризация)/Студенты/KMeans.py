# -*- coding: utf-8 -*-
"""
Created on Thu May 30 09:22:15 2024

@author: Sarum
"""

# Алгоритмы кластеризации используются при недостатке информации

from sklearn.cluster import KMeans
import pandas as pd
import seaborn as sns

#%%
df=pd.read_csv('students.csv', delimiter=',')
df.info()

#%%
df_cut =df[["Weight","Growth","Sex"]] 
df_cut.head()

df_cut = df_cut.dropna()

kmeans = KMeans(n_clusters=2, random_state=0)
kmeans.fit(df_cut[["Weight","Growth"]])
df_cut['Label']=kmeans.labels_

sns.scatterplot(data=df_cut,x="Weight",y="Growth", hue="Label")

#%%
sns.scatterplot(data=df_cut,x="Weight",y="Growth", hue="Sex")

#%% 
sum((df_cut['Sex']=='мужской')&(df_cut['Label']==1))

#%%
sum((df_cut['Sex']=='женский')&(df_cut['Label']==0))