# -*- coding: utf-8 -*-
"""
Created on Wed May 29 09:22:12 2024

@author: Sarum
"""

import pandas as pd

df = pd.read_csv("column_hell.csv")
print(df.head())

selected_columns = df.filter(like = "-")
print(selected_columns)