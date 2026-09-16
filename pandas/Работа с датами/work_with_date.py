# -*- coding: utf-8 -*-
"""
Created on Wed May 29 13:10:45 2024

@author: Sarum
"""

import pandas as pd

#%%
ether_grow = pd.read_csv('EtherSupplyGrowthChart.csv', parse_dates = ['Date(UTC)','UnixTimeStamp'])

ether_grow = ether_grow.rename(columns = {'Date(UTC)': 'date','UnixTimeStamp': 'unix', 'Value':'value'})

print(ether_grow.head())
print(ether_grow.info())

pd.to_datetime(ether_grow.unix,unit = 's')


def extract_data(df,column):
  df[column+'_year'] = df[column].apply(lambda x: x.year)
  df[column+'_month'] = df[column].apply(lambda x: x.month)
  
extract_data(ether_grow,'date')

ether_grow.index = ether_grow.date

#%%
print(ether_grow.head())

# среднее за месяц
print(ether_grow.resample("M"))

#заполнение недостающих данных
print(ether_grow.resample("D").ffill().head())

#%%
print(ether_grow['2016'].head())

print(ether_grow['2016-12'].head())

print(ether_grow['2016-01-01':'2016-12-31'].head())