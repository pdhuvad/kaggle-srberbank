# coding: utf-8
#!/usr/bin/python
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import gc
from scipy.stats import mode
from sklearn.decomposition import NMF
from sklearn import preprocessing
from sklearn.svm import SVR

#from sklearn.pipeline import Pipeline
#from sklearn.preprocessing import Imputer
#from sklearn.model_selection import cross_val_score

Train_Original=pd.read_csv('train.csv', parse_dates=['timestamp'])
Macro_Original=pd.read_csv('macro.csv',parse_dates=['timestamp'])
Train_Combined=pd.merge(Train_Original , Macro_Original,how='left',on='timestamp')

Test_Original=pd.read_csv('test.csv',parse_dates=['timestamp'])
Test_Combined=pd.merge(Test_Original ,Macro_Original ,how='left',on='timestamp')





print('combining rent price .....')  

business_rent=np.zeros(len(Train_Combined))
economy_rent=np.zeros(len(Train_Combined))


for index, row in Train_Combined.iterrows():
    #print(index)
    if row['num_room']==1 :
        business_rent[index]=row['rent_price_1room_bus']
        economy_rent[index]=row['rent_price_1room_eco']
    
    if row['num_room']==2 :
        business_rent[index]=row['rent_price_2room_bus']
        economy_rent[index]=row['rent_price_2room_eco']
    if row['num_room']==3 :
        business_rent[index]=row['rent_price_3room_bus']
        economy_rent[index]=row['rent_price_3room_eco']
    if row['num_room']>3 :
        business_rent[index]=row['rent_price_4+room_bus']
        economy_rent[index]=row['rent_price_4+room_bus']
    if np.isnan(row['num_room']) :
        business_rent[index]=np.mean([row['rent_price_1room_bus'], row['rent_price_2room_bus'], row['rent_price_3room_bus'], row['rent_price_4+room_bus']  ])
        economy_rent[index]=np.mean([row['rent_price_1room_eco'], row['rent_price_2room_eco'], row['rent_price_3room_eco']  ])


Train_Combined['eco_rent_combined']=economy_rent
Train_Combined['bus_rent_combined']=business_rent


del business_rent
del economy_rent

gc.collect()

business_rent=np.zeros(len(Test_Combined))
economy_rent=np.zeros(len(Test_Combined))


for index, row in Test_Combined.iterrows():
    #print(index)

    if row['num_room']==1 :
        business_rent[index]=row['rent_price_1room_bus']
        economy_rent[index]=row['rent_price_1room_eco']
    
    if row['num_room']==2 :
        business_rent[index]=row['rent_price_2room_bus']
        economy_rent[index]=row['rent_price_2room_eco']
    if row['num_room']==3 :
        business_rent[index]=row['rent_price_3room_bus']
        economy_rent[index]=row['rent_price_3room_eco']
    if row['num_room']>3 :
        business_rent[index]=row['rent_price_4+room_bus']
        economy_rent[index]=row['rent_price_4+room_bus']
    if np.isnan(row['num_room']) :
        business_rent[index]=np.mean([row['rent_price_1room_bus'], row['rent_price_2room_bus'], row['rent_price_3room_bus'], row['rent_price_4+room_bus']  ])
        economy_rent[index]=np.mean([row['rent_price_1room_eco'], row['rent_price_2room_eco'], row['rent_price_3room_eco']  ])


Test_Combined['eco_rent_combined']=economy_rent
Test_Combined['bus_rent_combined']=business_rent

             
# Save Target
y_train=Train_Combined['price_doc'].as_matrix()

print('selecting relevant columns.....')  
           
#new
features=Train_Combined.columns
use_fea=list(features[2:15])
# expand feature list

use_fea.append('railroad_km')
use_fea.append('cafe_count_5000')
use_fea.append('cafe_count_2000')
use_fea.append('metro_km_avto')
use_fea.append('metro_min_walk')
use_fea.append('bus_terminal_avto_km')
use_fea.append('big_market_km')
use_fea.append('oil_urals')
use_fea.append('mortgage_rate')
use_fea.append('unemployment')
use_fea.append('eco_rent_combined')
use_fea.append('bus_rent_combined')

train_fea=Train_Combined.columns
test_fea=Test_Combined.columns

drop_train_fea=[i for i in train_fea if i not in use_fea]
Train_Data=Train_Combined.drop(drop_train_fea,axis=1)

drop_test_fea=[i for i in test_fea if i not in use_fea]
Test_Data=Test_Combined.drop(drop_test_fea, axis=1)


#identify problemetic data



#S_life_sq1=Train_Data['life_sq'].isnull()
#S_life_sq2=Train_Data['life_sq'] < 5
#S_life_sq=S_life_sq1 | S_life_sq2
#
#
#S_maxfloor=Train_Data['max_floor'].isnull()
#S_material=Train_Data['material'].isnull()
#S_build_year=Train_Data['build_year'].isnull()
#S_num_room=Train_Data['num_room'].isnull()
#S_kitch_sq=Train_Data['kitch_sq'].isnull()
#S_state=Train_Data['state'].isnull()

#Train_Data1= Train_Data.copy(deep=True)
#Test_Data1= Test_Data.copy(deep=True)

print('Taking care of numeric data using NMF .....')  
drop_category_fea=['product_type', 'sub_area', 'build_year']
Train_Data1=Train_Data.drop(drop_category_fea, axis=1)
Test_Data1=Test_Data.drop(drop_category_fea, axis=1)
gc.collect()

#Train_Data1.fillna(Train_Data1.mean())
#Test_Data1.fillna(Test_Data1.mean())

numeric_fea=[]
for fea in use_fea:
    if fea not in drop_category_fea:
      numeric_fea.append(fea)
      Train_Data1[fea].fillna(Train_Data1[fea].median(), inplace=True)
      Test_Data1[fea].fillna(Test_Data1[fea].median(), inplace=True)



#dropped_df=tmp_df.drop(['modern_education_share','price_doc','timestamp','id'],axis=1)
#dropped_test_df=testm_df.drop(['modern_education_share','timestamp','id'],axis=1)
Data_FULLn_arr=[Train_Data1, Test_Data1]
DATA_Fulln=pd.concat(Data_FULLn_arr)
gc.collect()

X_Matrix=DATA_Fulln.as_matrix()
print('Doing NMF .....')       

nmf_model = NMF(n_components=5, random_state=1,
          alpha=.1, l1_ratio=.5).fit(X_Matrix)

H=nmf_model.components_
W=nmf_model.transform(X_Matrix)
X_transform=np.dot(W, H)



Data_FULL_arr=[Train_Data, Test_Data]
DATA_Full=pd.concat(Data_FULL_arr)
gc.collect()

print('Replacing nan with NMF values.....') 

for indx,fea in enumerate(numeric_fea):
        A1=pd.isnull(Train_Data[fea]).as_matrix()
        A2=pd.isnull(Test_Data[fea]).as_matrix()
        A=np.concatenate((A1, A2), axis=0)
        A3=A* X_transform[:,indx]
        X_Matrix[:,indx]=A3 + X_Matrix[:,indx]
        DATA_Full[fea]=X_Matrix[:,indx]
        
        
print('Taking care of build year .....')        
DATA_Full['build_year'].fillna(DATA_Full['build_year'].median(), inplace=True)
  
DATA_Full.loc[DATA_Full['build_year'] > 2020, 'build_year'] = 2000  
DATA_Full.loc[DATA_Full['build_year'] <1850, 'build_year'] = 1900 

DATA_Full_dummy=pd.get_dummies(DATA_Full)

X_full = DATA_Full_dummy.as_matrix()
X_full=np.nan_to_num(X_full)

#X_full = preprocessing.scale(X_full)


X_train=X_full[0:len(y_train), :]
X_test=X_full[len(y_train): , :]


n_samples = X_train.shape[0]
n_features = X_train.shape[1]


gc.collect()
print('starting model building using random forest.....')

# Estimate the score on the entire dataset, with no missing values
estimator = RandomForestRegressor(random_state=0, n_estimators=500, verbose=1)
estimator.fit(X_train, y_train)
predicted_y= estimator.predict(X_test)




#score = cross_val_score(estimator, X_full, y_full).mean()
#print("Score with the entire dataset = %.2f" % score)


AA=[list(Test_Combined['id']), predicted_y]
f=open('submit_NMF.csv', 'w')
f.write('id,price_doc\n')
for i,a in enumerate(AA[0]):
    f.write('{0},{1}\n'.format(AA[0][i], AA[1][i]))
f.close()

