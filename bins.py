# coding: utf-8
tmp_df=train_df
#!/usr/bin/python
import numpy as np
import pandas as pd
# coding: utf-8
bin1=[]
tmp=tmp1=tmp2=0
for i in range(11):
    tmp=(4-0)/11.
    tmp2=tmp1
    tmp1=tmp1+tmp
    
    tmp3=[tmp2,tmp1]
    bin1.append(tmp3)
    
bin2=[]
tmp=tmp1=tmp2=0
for i in range(16):
    tmp=(10-4)/16.
    if i==0:
        tmp2=tmp1=4
    else:
        tmp2=tmp1
    tmp1=tmp1+tmp
    
    tmp3=[tmp2,tmp1]
    bin2.append(tmp3)
    
bin3=[]
tmp=tmp1=tmp2=0
for i in range(16):
    tmp=(25-10)/16.
    if i==0:
        tmp2=tmp1=10
    else:
        tmp2=tmp1
    tmp1=tmp1+tmp
    
    tmp3=[tmp2,tmp1]
    bin3.append(tmp3)
    
bin4=[]
tmp=tmp1=tmp2=0
for i in range(8):
    tmp=(115-25)/8.
    if i==0:
        tmp2=tmp1=25
    else:
        tmp2=tmp1
    tmp1=tmp1+tmp
    
    tmp3=[tmp2,tmp1]
    bin4.append(tmp3)
    
bins=np.concatenate([bin1,bin2,bin3,bin4])
#bins[52]=np.array([75.625,86.875])
#bins=np.delete(bins,53,0)
tmp_df['price_class']=np.zeros(len(tmp_df))

tmp_df['price_class']=np.zeros(len(tmp_df))
tmp_df['price_class_max_freq']=np.zeros(len(tmp_df))
for i in range(len(bins)):
        tmp_df.loc[:,'price_class'][tmp_df['price_doc'].between(bins[i,0],bins[i,1],inclusive=True)]=i
        print(i)
        tmp_df.loc[:,'price_class_max_freq'][tmp_df['price_doc'].between(bins[i,0],bins[i,1],inclusive=True)]=tmp_df[tmp_df['price_doc'].between(bins[i,0],bins[i,1],inclusive=True)].price_doc.value_counts().idxmax()
map_class_price=tmp_df[['price_doc','price_class','price_class_max_freq']]
mapping_class_price=[]
for i in range(51):
    mapping_class_price.append([i,map_class_price[map_class_price.price_class==i].price_class_max_freq.max()])
