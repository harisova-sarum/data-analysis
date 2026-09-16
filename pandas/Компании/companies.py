# -*- coding: utf-8 -*-
"""
Created on Wed May 29 17:23:15 2024

@author: Sarum
"""

import pandas as pd
df = pd.read_csv('companies.csv', sep = ';')

#%%
df.head()

#%%
import numpy as np
df.groupby('company', as_index=True).agg({'income': np.mean})