# -*- coding: utf-8 -*-
"""
Created on Wed May 29 09:49:24 2024

@author: Sarum
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('income.csv')
print(df.head())

df.income.plot()
df['income'].plot()
df.plot()
df.plot(kind='line')
sns.lineplot(data=df)
sns.lineplot(x=df.index, y=df.income)
plt.plot(df.index, df.income)