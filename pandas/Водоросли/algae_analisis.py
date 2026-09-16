# -*- coding: utf-8 -*-
"""
Created on Wed May 29 09:39:05 2024

@author: Sarum
"""
#%%
import pandas as pd
import numpy as np

concentrations = pd.read_csv('algae.csv')
concentrations.info()

#%%
concentrations.head()

#%%
mean_concentrations = concentrations.groupby('genus')\
  .aggregate({'sucrose': 'mean', 'alanin': 'mean', 'citrate': 'mean', 'glucose': 'mean', 'oleic_acid': 'mean'})
print(mean_concentrations)

#%%
alain_of_fucus = concentrations.query("genus == 'Fucus'").alanin
print(alain_of_fucus)

#%%
concentrations.groupby('group').aggregate({'sucrose': lambda x: x.max() - x.min(), 'citrate': 'var', 'species': 'count'})