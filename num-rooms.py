# coding: utf-8

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

train_org_df=pd.read_csv('train.csv',parse_dates=['timestamp'])
macro_org_df=pd.read_csv('macro.csv',parse_dates=['timestamp'])
train_df=pd.merge(train_org_df,macro_org_df,how='left',on='timestamp')
column_list=list(train_df.columns)
column_list


def room_rent_drop(df):
    room_rent_df.ix[room_rent_df['num_room']== 1,['rent_price_4+room_bus', 'rent_price_3room_bus', 'rent_price_2room_bus',  'rent_price_3room_eco', 'rent_price_2room_eco', ]]=0
    room_rent_df.ix[room_rent_df['num_room']== 2,['rent_price_4+room_bus', 'rent_price_3room_bus',  'rent_price_1room_bus', 'rent_price_3room_eco',  'rent_price_1room_eco']]=0
    room_rent_df.ix[room_rent_df['num_room']== 3,['rent_price_4+room_bus',  'rent_price_2room_bus', 'rent_price_1room_bus', 'rent_price_2room_eco', 'rent_price_1room_eco']]=0
    room_rent_df.ix[room_rent_df['num_room'] >= 4,[ 'rent_price_3room_bus', 'rent_price_2room_bus', 'rent_price_1room_bus', 'rent_price_3room_eco', 'rent_price_2room_eco', 'rent_price_1room_eco']]=0
    
    return df


nan_room_df=train_df[train_df.num_room.isnull()]
nan_room_df['price_doc'].max()
nan_room_df['price_doc'].min()
111111112/190000
nan_room_df['price_doc'].median()
nan_room_df['price_doc'].mean()
plt.hist(nan_room_df['price_doc'],bins=range(190000,111111112+190000 , 190000))
plt.show()
