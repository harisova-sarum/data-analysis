# -*- coding: utf-8 -*-
"""
Created on Wed May 29 09:24:26 2024

@author: Sarum
"""
# %%
import pandas as pd
import numpy as np

dota2 = pd.read_csv('dota_hero_stats.csv')
print(dota2.head())
#%%
dota2.columns

# %%
dota2.groupby(['legs']).aggregate({'legs': 'count'})

#%%
dota2.groupby(['attack_type','primary_attr']).count()

#%%
dota2.roles
#%% 
number_of_roles = [len(i.split(',')) for i in dota2.roles]


dota2 = dota2.assign(roles_count = pd.Series(number_of_roles))

dota2.roles_count.hist()