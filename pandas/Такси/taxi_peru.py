# -*- coding: utf-8 -*-
"""
Created on Wed May 29 17:34:41 2024

@author: Sarum
"""

import pandas as pd
taxi = pd.read_csv('taxi_peru.csv', sep = ';', parse_dates=['start_at','end_at','arrived_at'])
#%%
taxi.head()

#%%
taxi.columns

#%%
s = taxi.groupby('source').count()
s.head()

#%%
s.sort_values('journey_id', ascending = False)

#%%
import matplotlib.pyplot as plt
import numpy as np
source = (s.sort_values('journey_id', ascending = False).index.to_numpy())
y_pos = np.arange(len(source))
counts = s.sort_values('journey_id', ascending = False)['journey_id']

plt.bar(y_pos, counts, align='center', alpha=0.5)
plt.xticks(y_pos, source)
plt.ylabel('Count')
plt.xlabel('Platform')
plt.show()

#%%
taxi['icon']
#%%
import seaborn as sns

sns.countplot(taxi['icon'])

#%%
plt.figure(figsize = (12,9))
sns.countplot(taxi['end_state'], hue = taxi['source'])

#%%
plt.figure(figsize = (12,9))
sns.countplot(taxi['driver_score'])

#%%
d_s = taxi.groupby('driver_score')\
  .count()\
  .sort_values('driver_score')
d_s['percentage'] = (d_s['journey_id']/d_s['journey_id'].sum()*100).round(2)
driver_score_counts = d_s[['percentage']].reset_index()
#print(driver_score_counts)

ax = sns.barplot(x='driver_score', y='percentage', data=driver_score_counts, color='blue', alpha=0.5)
ax.set(xlabel='Driver score', ylabel='Percentage')
sns.despine() # убрать часть рамки графика
plt.show()

#%%
d_s = taxi.groupby('rider_score')\
  .count()\
  .sort_values('rider_score')
d_s['percentage'] = (d_s['journey_id']/d_s['journey_id'].sum()*100).round(2)
driver_score_counts = d_s[['percentage']].reset_index()
#print(driver_score_counts)


ax = sns.barplot(x='rider_score', y='percentage', data=driver_score_counts, color='blue', alpha=0.5)
ax.set(xlabel='Driver score', ylabel='Percentage')
sns.despine() # убрать часть рамки графика
plt.show()