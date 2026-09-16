# -*- coding: utf-8 -*-
"""
Created on Thu May 30 09:22:15 2024

@author: Sarum
"""

# Предсказание числового признака

import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


df=pd.read_csv('students.csv', delimiter=',')


#%% Гипотеза: рост линейно зависит от размера обуви
df_cut=df[["Growth", "Shoe size"]]
df_cut=df_cut.dropna()
sns.scatterplot(data=df_cut,x="Shoe size",y="Growth")

#%%
X_train, X_test, y_train, y_test = train_test_split(df[['Shoe size']], df_cut[['Growth']], test_size=0.3, random_state=42)

linearRegression = LinearRegression()
results = linearRegression.fit(X_train.values.reshape(-1,1),y_train.values)

print(results.coef_, results.intercept_)

y_pred=results.predict(X_test.values.reshape(-1,1))

print(df_cut.head(20))

from sklearn.metrics import mean_absolute_error
mean_absolute_error(y_pred, y_test)

#%% Гипотеза: длина указательного пальца линейно зависит от длины среднего пальца и мизинца
df.info()
df_cut = df[["MIddle and index finger","Middle and ring finger","Middle and little finger"]]
df_cut=df_cut.dropna()

X_train, X_test, y_train, y_test = train_test_split(df_cut[["Middle and ring finger","Middle and little finger"]], df_cut[["MIddle and index finger"]], test_size=0.3, random_state=42)

results = linearRegression.fit(X_train.values.reshape(-1,2),y=y_train.values)

print(results.coef_, results.intercept_)

y_pred=results.predict(X_test.values.reshape(-1,2))
mean_absolute_error(y_pred, y_test)



