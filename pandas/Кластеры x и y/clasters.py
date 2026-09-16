# -*- coding: utf-8 -*-
"""
Created on Wed May 29 09:54:31 2024

@author: Sarum
"""

import pandas as pd
import seaborn as sns


df = pd.read_csv('dataset_209770_6 (1).txt', sep=' ')
print(df.head())

sns.scatterplot(x = 'x', y = 'y', data = df)