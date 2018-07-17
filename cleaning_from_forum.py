# coding: utf-8
# %load num-rooms.py
get_ipython().magic('reset')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

train_org_df=pd.read_csv('train.csv',parse_dates=['timestamp'])
train_org_df.price_doc = train_org_df.price_doc/1000000.
macro_org_df=pd.read_csv('macro.csv',parse_dates=['timestamp'])
train_df=pd.merge(train_org_df,macro_org_df,how='left',on='timestamp')
test_org_df=pd.read_csv('test.csv',parse_dates=['timestamp'])
test_df=pd.merge(test_org_df,macro_org_df,on='timestamp',how='left')
test=test_df.copy()
train=train_df.copy()

id_test = test.id

#multiplier = 0.969

#clean data
bad_index = train[train.life_sq > train.full_sq].index
train.ix[bad_index, "life_sq"] = np.NaN
equal_index = [601,1896,2791]
test.ix[equal_index, "life_sq"] = test.ix[equal_index, "full_sq"]
bad_index = test[test.life_sq > test.full_sq].index
test.ix[bad_index, "life_sq"] = np.NaN
bad_index = train[train.life_sq < 5].index
train.ix[bad_index, "life_sq"] = np.NaN
bad_index = test[test.life_sq < 5].index
test.ix[bad_index, "life_sq"] = np.NaN
bad_index = train[train.full_sq < 5].index
train.ix[bad_index, "full_sq"] = np.NaN
bad_index = test[test.full_sq < 5].index
test.ix[bad_index, "full_sq"] = np.NaN
kitch_is_build_year = [13117]
train.ix[kitch_is_build_year, "build_year"] = train.ix[kitch_is_build_year, "kitch_sq"]
bad_index = train[train.kitch_sq >= train.life_sq].index
train.ix[bad_index, "kitch_sq"] = np.NaN
bad_index = test[test.kitch_sq >= test.life_sq].index
test.ix[bad_index, "kitch_sq"] = np.NaN
bad_index = train[(train.kitch_sq == 0).values + (train.kitch_sq == 1).values].index
train.ix[bad_index, "kitch_sq"] = np.NaN
bad_index = test[(test.kitch_sq == 0).values + (test.kitch_sq == 1).values].index
test.ix[bad_index, "kitch_sq"] = np.NaN
bad_index = train[(train.full_sq > 210) & (train.life_sq / train.full_sq < 0.3)].index
train.ix[bad_index, "full_sq"] = np.NaN
bad_index = test[(test.full_sq > 150) & (test.life_sq / test.full_sq < 0.3)].index
test.ix[bad_index, "full_sq"] = np.NaN
bad_index = train[train.life_sq > 300].index
train.ix[bad_index, ["life_sq", "full_sq"]] = np.NaN
bad_index = test[test.life_sq > 200].index
test.ix[bad_index, ["life_sq", "full_sq"]] = np.NaN
train.product_type.value_counts(normalize= True)
test.product_type.value_counts(normalize= True)
bad_index = train[train.build_year < 1500].index
train.ix[bad_index, "build_year"] = np.NaN
bad_index = test[test.build_year < 1500].index
test.ix[bad_index, "build_year"] = np.NaN
bad_index = train[train.num_room == 0].index 
train.ix[bad_index, "num_room"] = np.NaN
bad_index = test[test.num_room == 0].index 
test.ix[bad_index, "num_room"] = np.NaN
bad_index = [10076, 11621, 17764, 19390, 24007, 26713, 29172]
train.ix[bad_index, "num_room"] = np.NaN
bad_index = [3174, 7313]
test.ix[bad_index, "num_room"] = np.NaN
bad_index = train[(train.floor == 0).values * (train.max_floor == 0).values].index
train.ix[bad_index, ["max_floor", "floor"]] = np.NaN
bad_index = train[train.floor == 0].index
train.ix[bad_index, "floor"] = np.NaN
bad_index = train[train.max_floor == 0].index
train.ix[bad_index, "max_floor"] = np.NaN
bad_index = test[test.max_floor == 0].index
test.ix[bad_index, "max_floor"] = np.NaN
bad_index = train[train.floor > train.max_floor].index
train.ix[bad_index, "max_floor"] = np.NaN
bad_index = test[test.floor > test.max_floor].index
test.ix[bad_index, "max_floor"] = np.NaN
train.floor.describe(percentiles= [0.9999])
bad_index = [23584]
train.ix[bad_index, "floor"] = np.NaN
train.material.value_counts()
test.material.value_counts()
train.state.value_counts()
bad_index = train[train.state == 33].index
train.ix[bad_index, "state"] = np.NaN
test.state.value_counts()


test_df=test.copy()
train_df=train.copy()
def knn1_impute(train_org_df):
    #df=train_org_df[['full_sq','life_sq','num_room', 'price_doc']]
    df=train_org_df[['full_sq','life_sq','num_room','sub_area', 'price_doc']]
    df=df[np.isfinite(df['full_sq'])]
    df=pd.get_dummies(df)
    #seperate test and train for KNN1 for which life_sq is never null
    train_df = df[(df.num_room.notnull()) & (df.life_sq.notnull())]
    train_X_df=train_df.drop('num_room',1)
    train_y_df=train_df.num_room
    train_X = train_X_df
    train_y = train_y_df
    #from sklearn.model_selection import train_test_split
    #train_X, test_X, train_y, test_y = train_test_split(train_X_df,train_y_df,test_size=.2,random_state=16)
    hidden_df=df[df.num_room.isnull() & (df.life_sq.notnull())]
    hidden_X_df=hidden_df.drop('num_room',1)

    from sklearn.neighbors import KNeighborsClassifier
    knn=KNeighborsClassifier()
    knn.fit(train_X,train_y)

    #test_predict_y=knn.predict(test_X)
    #knn.score(test_X,test_y)
    hidden_X_df['num_room']=knn.predict(hidden_X_df)

    tmp1=hidden_X_df.num_room
    tmp1.shape
    imputed_train_df=train_org_df.copy()
    imputed_train_df.update(tmp1,raise_conflict=True)
    return imputed_train_df
knn1_train_df=knn1_impute(train_df)
def knn2_impute(train_org_df):
    df=train_org_df[['full_sq','life_sq','kitch_sq','num_room','sub_area', 'price_doc']]
    df=pd.get_dummies(df)

    #seperate test and train for KNN1 for which life_sq is never null

    train_df = df[(df.num_room.notnull())]
    train_X_df=train_df.drop(['num_room','life_sq','kitch_sq'],1)
    train_y_df=train_df.num_room
    train_X = train_X_df
    train_y = train_y_df
    #from sklearn.model_selection import train_test_split
    #train_X, test_X, train_y, test_y = train_test_split(train_X_df,train_y_df,test_size=.2,random_state=16)
    hidden_df=df[df.num_room.isnull() & (df.life_sq.isnull()) & (df.kitch_sq.isnull())]
    hidden_X_df=hidden_df.drop(['num_room','life_sq','kitch_sq'],1)

    from sklearn.neighbors import KNeighborsClassifier
    knn=KNeighborsClassifier()
    knn.fit(train_X,train_y)

    #test_predict_y=knn.predict(test_X)
    #knn.score(test_X,test_y)
    hidden_X_df['num_room']=knn.predict(hidden_X_df)

    tmp1=hidden_X_df.num_room
    imputed_train_df=train_org_df.copy()
    imputed_train_df.update(tmp1,raise_conflict=True)
    return imputed_train_df
knn2_train_df=knn2_impute(knn1_train_df)
def knn2_impute(train_org_df):
    df=train_org_df[['full_sq','life_sq','kitch_sq','num_room','sub_area', 'price_doc']]
    df=df[np.isfinite(df['full_sq'])]
    df=pd.get_dummies(df)

    #seperate test and train for KNN1 for which life_sq is never null

    train_df = df[(df.num_room.notnull())]
    train_X_df=train_df.drop(['num_room','life_sq','kitch_sq'],1)
    train_y_df=train_df.num_room
    train_X = train_X_df
    train_y = train_y_df
    #from sklearn.model_selection import train_test_split
    #train_X, test_X, train_y, test_y = train_test_split(train_X_df,train_y_df,test_size=.2,random_state=16)
    hidden_df=df[df.num_room.isnull() & (df.life_sq.isnull()) & (df.kitch_sq.isnull())]
    hidden_X_df=hidden_df.drop(['num_room','life_sq','kitch_sq'],1)

    from sklearn.neighbors import KNeighborsClassifier
    knn=KNeighborsClassifier()
    knn.fit(train_X,train_y)

    #test_predict_y=knn.predict(test_X)
    #knn.score(test_X,test_y)
    hidden_X_df['num_room']=knn.predict(hidden_X_df)

    tmp1=hidden_X_df.num_room
    imputed_train_df=train_org_df.copy()
    imputed_train_df.update(tmp1,raise_conflict=True)
    return imputed_train_df
knn2_train_df=knn2_impute(knn1_train_df)
train_df=knn2_train_df.copy()
def room_rent_drop(df):
    room_rent_df = df
    room_rent_df.ix[room_rent_df['num_room']== 1,['rent_price_4+room_bus', 'rent_price_3room_bus', 'rent_price_2room_bus',  'rent_price_3room_eco', 'rent_price_2room_eco', ]]=0
    room_rent_df.ix[room_rent_df['num_room']== 2,['rent_price_4+room_bus', 'rent_price_3room_bus',  'rent_price_1room_bus', 'rent_price_3room_eco',  'rent_price_1room_eco']]=0
    room_rent_df.ix[room_rent_df['num_room']== 3,['rent_price_4+room_bus',  'rent_price_2room_bus', 'rent_price_1room_bus', 'rent_price_2room_eco', 'rent_price_1room_eco']]=0
    room_rent_df.ix[room_rent_df['num_room'] >= 4,[ 'rent_price_3room_bus', 'rent_price_2room_bus', 'rent_price_1room_bus', 'rent_price_3room_eco', 'rent_price_2room_eco', 'rent_price_1room_eco']]=0
    
    return room_rent_df
def room_rent_avgs_only(df):
    train_df=room_rent_drop(df)
    tmp1 =train_df[train_df.num_room.isnull()]
    tmp1['rent_eco_all']=0
    tmp1.ix[tmp1.num_room.isnull(),'rent_eco_all'] = tmp1[['rent_price_2room_eco', 'rent_price_1room_eco', 'rent_price_3room_eco']].mean(axis=1)
    tmp2 =train_df[train_df.num_room.notnull()]
    tmp2['rent_eco_all']=tmp2[['rent_price_2room_eco', 'rent_price_1room_eco', 'rent_price_3room_eco']].sum(axis=1)
    train_df =pd.concat([tmp1,tmp2],ignore_index=True)
    tmp3 =train_df[train_df.num_room.isnull()]
    tmp3['rent_bus_all']=0
    tmp3.ix[tmp3.num_room.isnull(),'rent_eco_all'] = tmp3[['rent_price_2room_bus', 'rent_price_1room_bus', 'rent_price_3room_bus','rent_price_4+room_bus']].mean(axis=1)
    tmp4 =train_df[train_df.num_room.notnull()]
    tmp4['rent_bus_all']=tmp4[['rent_price_2room_bus', 'rent_price_1room_bus', 'rent_price_3room_bus','rent_price_4+room_bus']].sum(axis=1)
    tmp4.tail()
    train_df =pd.concat([tmp3,tmp4],ignore_index=True)
    return train_df
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Imputer
from sklearn.model_selection import cross_val_score
train_df=room_rent_avgs_only(train_df)
test_df=room_rent_avgs_only(test_df)
tmp10 =train_df.copy()
tmp10.ix[tmp10.num_room>=4,'rent_eco_all']= tmp10['rent_bus_all']
train_df = tmp10.copy()
rng = np.random.RandomState(0)
tmp_df=train_df.copy()
y_train=train_df['price_doc'].as_matrix()

#new
features=train_df.columns
use_fea=list(features[2:15])
# expand
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
use_fea.append('rent_bus_all')
use_fea.append('rent_eco_all')
train_fea=train_df.columns
test_fea=test_df.columns

drop_train_fea=[i for i in train_fea if i not in use_fea]
dropped_df=tmp_df.drop(drop_train_fea,axis=1)

drop_test_fea=[i for i in test_fea if i not in use_fea]
dropped_test_df=test_df.drop(drop_test_fea, axis=1)


#dropped_df=tmp_df.drop(['modern_education_share','price_doc','timestamp','id'],axis=1)
#dropped_test_df=testm_df.drop(['modern_education_share','timestamp','id'],axis=1)
Data_FULL_arr=[dropped_df, dropped_test_df]
DATA_Full=pd.concat(Data_FULL_arr)


DATA_Full_dummy=pd.get_dummies(DATA_Full)
X_full = DATA_Full_dummy.as_matrix()
X_full=np.nan_to_num(X_full)

X_train=X_full[0:len(y_train), :]
X_test=X_full[len(y_train): , :]

#y_full=dataset[:,[291]]
#relevent_columns=[i for i in range(2,291)]
#relevent_columns1=[i for i in range(292,391)]
#rel_col=np.concatenate([relevent_columns,relevent_columns1],axis=0)
#X_full=dataset[:,rel_col]

n_samples = X_train.shape[0]
n_features = X_train.shape[1]




# Estimate the score on the entire dataset, with no missing values
estimator = RandomForestRegressor(random_state=0, n_estimators=500, verbose=1)
estimator.fit(X_train, y_train)
predicted_y= estimator.predict(X_test)
#score = cross_val_score(estimator, X_full, y_full).mean()
#print("Score with the entire dataset = %.2f" % score)


AA=[list(test_df['id']), predicted_y]
f=open('submit7.csv', 'w')
f.write('id,price_doc\n')
for i,a in enumerate(AA[0]):
    f.write('{0},{1}\n'.format(AA[0][i], AA[1][i]))
f.close()
