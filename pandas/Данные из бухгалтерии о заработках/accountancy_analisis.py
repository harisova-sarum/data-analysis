# -*- coding: utf-8 -*-
"""
Created on Wed May 29 09:28:35 2024

@author: Sarum
"""
#%%

import pandas as pd
import numpy as np

incomes = pd.read_csv('accountancy.csv')
incomes.head()

#%%
incomes.info()

#%%
incomes.groupby(['Executor', 'Type']).aggregate({'Salary': 'mean'})