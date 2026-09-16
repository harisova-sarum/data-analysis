# -*- coding: utf-8 -*-
"""
Created on Wed May 29 09:57:04 2024

@author: Sarum
"""

import pandas as pd
import seaborn as sns

genomic_distances = pd.read_csv('genome_matrix.csv')
print(genomic_distances.head())

genomic_distances.index = genomic_distances['Unnamed: 0']

g = sns.heatmap(genomic_distances.iloc[:,1:], cmap="viridis")
g.xaxis.set_ticks_position('top')
g.xaxis.set_tick_params(rotation=90)