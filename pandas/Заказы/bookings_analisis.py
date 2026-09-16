# -*- coding: utf-8 -*-
"""
Created on Wed May 29 17:14:34 2024

@author: Sarum
"""

import pandas as pd
bookings = pd.read_csv('bookings.csv', sep = ';')
bookings.columns = bookings.columns.str.replace(' ', '_')\
  .str.lower()
print(bookings.shape)
print(bookings.info())

#%%
bookings.groupby('country', as_index = False)\
  .agg({'is_canceled': sum}) \
  .sort_values('is_canceled',ascending = False)[:5]

#%%
bookings.columns

#%%
round(bookings['stays_total_nights'][bookings['hotel']=='City Hotel'].mean(),2)

#%%
round(bookings['stays_total_nights'][bookings['hotel']=='Resort Hotel'].mean(),2)

#%%
bookings[bookings['reserved_room_type']!=bookings['assigned_room_type']].shape[0]

#%%
import datetime as dt
bookings['reservation_status_date'] = pd.to_datetime(bookings['reservation_status_date'])

bookings['reservation_status_date']\
            .dt.month[bookings['reservation_status_date']\
            .dt.year==2016]\
            .value_counts()

#%%
bookings['reservation_status_date']\
            .dt.month[bookings['reservation_status_date']\
            .dt.year==2017]\
            .value_counts()
            
#%%
bookings.query("is_canceled==1")\
      .query("hotel=='City Hotel'")\
      .groupby('arrival_date_year')['arrival_date_month'].value_counts()
      
#%%
bookings[['adults','children','babies']].mean()

#%%
bookings['total_kids'] = bookings['children'] + bookings['babies']
round(bookings.groupby("hotel")['total_kids'].mean(),2)

#%%
bookings['has_kids'] = bookings['total_kids'] > 0
bookings.groupby("has_kids")['is_canceled'].value_counts()
