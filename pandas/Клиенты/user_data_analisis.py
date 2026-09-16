# -*- coding: utf-8 -*-
"""
Created on Wed May 29 18:04:22 2024

@author: Sarum
"""

import pandas as pd
user_data = pd.read_csv('user_data.csv')
logs = pd.read_csv('logs.csv')

#%%
user_data.shape

#%%
user_data.describe()

#%%
user_data.info()

#%%
logs.shape

#%%
logs.describe()

#%%
logs.info()

#%%
logs.platform.unique()

#%% Какой клиент совершил больше всего успешных операций?
success_number = logs.query('success == True')\
          .groupby('platform')\
          .agg({'success': 'sum'}).sort_values('success', ascending = False)
maxi = success_number.max()[0]
result = success_number.query('success == @maxi').index.to_list()
sorted(result)

success_number = logs.query('success == True')\
          .groupby('client')\
          .agg({'success': 'sum'}).sort_values('success', ascending = False)
maxi = success_number.max()[0]
result = success_number.query('success == @maxi').index.to_list()
sorted(result)
#%% Какая платформа наиболее популярна среди премиумных клиентов?
import seaborn as sns
all_info = user_data.merge(logs)
print(all_info)
all_info.groupby('platform')\
          .agg({'premium': 'count'})
          
sns.barplot(x='age', y='premium', data=all_info, color='blue', alpha=0.5)

import matplotlib.pyplot as plt
sns.color_palette('tab10')
sns.displot(user_data, x="age",  hue="premium")
sns.displot(user_data, x="age",  hue="premium", kind="kde")

succsess_number = logs.groupby('client', as_index = False).agg({'success': 'count'}).sort_values('success')
print(succsess_number)

client_succsess_number = succsess_number.groupby('success', as_index = False).agg({'client': 'count'})
client_succsess_number.columns
print(client_succsess_number)

plt.figure(figsize = (32,28))
sns.displot(client_succsess_number)

success_number = all_info.query('success == True')\
          .query("platform == 'computer'")
print(success_number)

plt.figure(figsize=(12, 8))
sns.countplot(success_number.age)
