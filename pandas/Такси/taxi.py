# -*- coding: utf-8 -*-
"""
Created on Wed May 29 17:02:31 2024

@author: Sarum
"""

import pandas as pd

taxi = pd.read_csv('2_taxi_nyc.csv')
#%%
taxi.dtypes
#%%
taxi.head()
#%%
taxi.pickups.sum()
#%%
pickups_by_mon_bor = taxi.groupby(['borough','pickup_month'], as_index = False)\
  .agg({'pickups': sum}) \
  .sort_values('pickups',ascending = False)[['pickup_month', 'borough', 'pickups']]
print(pickups_by_mon_bor)

#%%
def temp_to_celcius(x):
  return (x - 32)*5/9

taxi['temp_C'] = temp_to_celcius(taxi['temp'])
taxi['temp_C'][:5]