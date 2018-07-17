# coding: utf-8
# %load multiclass_ensemble.py
get_ipython().magic('reset')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

train_org_df=pd.read_csv('train.csv',parse_dates=['timestamp'])
update_train=pd.read_csv('BAD_ADDRESS_FIX.csv')
train_org_df.update(update_train,overwrite=True)
train_org_df.price_doc = train_org_df.price_doc/1000000.
macro_org_df=pd.read_csv('macro.csv',parse_dates=['timestamp'])
train_df=pd.merge(train_org_df,macro_org_df,how='left',on='timestamp')
test_org_df=pd.read_csv('test.csv',parse_dates=['timestamp'])
test_df=pd.merge(test_org_df,macro_org_df,on='timestamp',how='left')

# move price_doc to last
price_df=train_df.price_doc
train_df=train_df.drop('price_doc', axis=1)
train_df['price_doc']=price_df

# wrong build year update from 1691 to 1991
train_df.set_value(26332,'build_year',1991);
train_df.set_value(30275,'build_year',1971);
train_df.set_value(30150,'build_year',2015);
train_df.set_value(10089,'build_year',2007);
train_df.set_value(10089,'state',3);
train_df.set_value(15220,'build_year',1965); # was 4965
train_df.set_value(13992,'build_year',2017) ; # was 20
test_df.set_value(2995,'build_year',2015);
#train_df.full_sq.set_value(4678,'full_sq',65); ### doubt anni daa
#train_df.set_value(27460,'price_doc', 7.1249624) 
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Imputer
from sklearn.model_selection import cross_val_score
from sklearn import svm
print("cases initilized")
global case_1
global case_2
global case_3
global case_4

class Predict_Class():
    #_data = pd.DataFrame
    def __init__(self, train_rows):
        self.highcutoff = 100
        self.lowpricecutoff = 2.1
        sa=train_rows[train_rows.full_sq < self.highcutoff];
        
        s3=sa[sa.price_doc>self.lowpricecutoff]
        s4=sa[sa.price_doc <= self.lowpricecutoff]

        t3=s3.price_doc
        t4=s4.price_doc
        
        s3=s3.drop(['price_doc','sub_area', 'full_sq' ,'product_type'],axis=1)
        s3=s3.iloc[:,2:20]
        for fea in s3.columns:
                s3[fea].fillna(s3[fea].median(), inplace=True)
        for fea in s3.columns:
                s3[fea].fillna(0, inplace=True)
        s4=s4.drop(['price_doc','sub_area', 'full_sq','product_type'],axis=1)
        s4=s4.iloc[:,2:20]
        for fea in s4.columns:
                s4[fea].fillna(s4[fea].median(), inplace=True)
        for fea in s4.columns:
                s4[fea].fillna(0, inplace=True)
        
                
        d3=np.ones(len(t3))
        d4=np.ones(len(t4))*-1
        print ("s3 shape",s3.shape)
        X = pd.concat([s4,s4,s3,s4,s3,s4,s4,s4,s4])
        y = np.concatenate([d4,d4,d3,d4,d3,d4,d4,d4,d4],axis=0)
        self.y=y
        self.X=X
        #print (X)
        #self.decision_clf = svm.SVC()
        self.decision_clf = RandomForestClassifier()
        self.decision_clf.fit(X, y)
        
        
    def predict(self,test_row):
                pclass=self.decision_clf.predict(test_row)[0]
                return pclass


selectedRows=(train_df.sub_area=='Kapotnja') |              (train_df.sub_area=='Poselenie Sosenskoe') 

TwoClassdata = train_df[selectedRows]

TwoClassModel=Predict_Class(TwoClassdata)
class Poselenie_Sosenskoe():
    #_data = pd.DataFrame
    def __init__(self, train_rows, twoclass):
        self.TwoClassModel=twoclass
        self.highcutoff = 250
        self.lowpricecutoff = 2.1
        #self.train_target=train_rows.price_doc
        #train_rows=train_rows.drop(['price_doc','sub_area','product_type'],axis=1)
        #train_rows1=train_rows.iloc[:,2:20]
        #self.train=train_rows1
        s1=train_rows[train_rows.full_sq > self.highcutoff];
        sa=train_rows[train_rows.full_sq < self.highcutoff];
        s2=sa[(sa.state.isnull()) & (sa.life_sq.isnull())]
        s3a = sa[~((sa.state.isnull()) & (sa.life_sq.isnull()))]
        s3=s3a[s3a.price_doc>self.lowpricecutoff]
        s4=s3a[s3a.price_doc <= self.lowpricecutoff]
        ta=sa.price_doc
        t1=s1.price_doc
        t2=s2.price_doc
        t3=s3.price_doc
        t4=s4.price_doc
        s1=s1.drop(['price_doc','sub_area','product_type'],axis=1)
        
        # HighCutoff
        s1=s1.iloc[:,2:20]
        for fea in s1.columns:
                s1[fea].fillna(s1[fea].median(), inplace=True)
        for fea in s1.columns:
                s1[fea].fillna(0, inplace=True)
        # s2 ~HighCutoff NaN        
        s2=s2.drop(['price_doc','sub_area','product_type'],axis=1)
        s2=s2.iloc[:,2:20]
        for fea in s2.columns:
                s2[fea].fillna(s2[fea].median(), inplace=True)
        for fea in s2.columns:
                s2[fea].fillna(0, inplace=True)
         # s3 ~HighCutoff ~NaN >LC       
        s3=s3.drop(['price_doc','sub_area','product_type'],axis=1)
        s3=s3.iloc[:,2:20]
        for fea in s3.columns:
                s3[fea].fillna(s3[fea].median(), inplace=True)
        for fea in s3.columns:
                s3[fea].fillna(0, inplace=True)
                
        # s4 ~HighCutoff ~NaN < LC
        s4=s4.drop(['price_doc','sub_area','product_type'],axis=1)
        s4=s4.iloc[:,2:20]
        for fea in s4.columns:
                s4[fea].fillna(s4[fea].median(), inplace=True)
        for fea in s4.columns:
                s4[fea].fillna(0, inplace=True)
        
        ## What is sa used for??????
        sa=sa.drop(['price_doc','sub_area','product_type'],axis=1)
        sa=sa.iloc[:,2:20]
        for fea in sa.columns:
                sa[fea].fillna(sa[fea].median(), inplace=True)
        for fea in sa.columns:
                sa[fea].fillna(0, inplace=True)
        ######### ESTIMATOR SELECTION CLASSIFIER  #############        
        d3=np.ones(len(t3))
        d4=np.ones(len(t4))*-1
        print ("s3 shape",s3.shape)
        X = pd.concat([s3,s4,s3,s4])
        y = np.concatenate([d3,d4,d3,d4],axis=0)
        
        from sklearn.utils import shuffle
        X, y = shuffle(X, y, random_state=0)
        self.y=y
        self.X=X
        #print (X)
        self.decision_clf = svm.SVC()
        #self.decision_clf = RandomForestClassifier()
        self.decision_clf.fit(X, y)
        
        ########## ESTIMATOR 2 ########## 
        estimator2r = RandomForestRegressor(random_state=0, n_estimators=500, verbose=1)
        estimator2l = LinearRegression()

        estimator2r.fit(s3, t3)
        self.estimator2r=estimator2r
        
        
        s2r=s3.full_sq
        s2r=s2r.reshape(len(s2r),1)
        t2r=t3
        t2r=t2r.reshape(len(t2r),1)   
        estimator2l.fit(s2r, t2r)
        self.estimator2l=estimator2l
    
        ########### ESTIMATOR 3 ##########
        ##estimator3 = RandomForestRegressor(random_state=0, n_estimators=500)
        #estimator3=LinearRegression()
   #
        #s3r=s3.full_sq
        #s3r=s3r.reshape(len(s3r),1)
        #t3r=t3
        #t3r=t3r.reshape(len(t3r),1)   
        #estimator3.fit(s3r, t3r)
        #
        #
        #self.estimator3=estimator3
        #
        ##print (type(estimator))
        #estimator3.fit(s3, t3)
        ##estimator3.fit(sa, ta)
        #self.estimator3=estimator3
        
        
        self.s1=s1
        self.t1=t1
        self.s2=s2
        self.t2=t2
        self.s3=s3
        self.t3=t3
        self.s4=s4
        self.t4=t4

        
        self.counts2_pred=0

    def drop_test(test_df):
        
        return test_df
    def predict_price(self,test_row):
        test_row1=test_row
        flag=0
        if ((test_row.state.isnull()) & (test_row.life_sq.isnull())).iloc[0]:
            flag=1
        test_row=test_row.drop(['sub_area','product_type'],axis=1)
        test_row=test_row.iloc[:,2:20]
        for fea in test_row.columns:
            test_row[fea].fillna(test_row[fea].median(),inplace=True)
        for fea in test_row.columns:
            test_row[fea].fillna(0,inplace=True)
        dum=test_row.iloc[0]['full_sq']
        #print (dum)
        
          

        if dum > self.highcutoff:
            x1 = test_row.full_sq
            x1 = x1.reshape(len(x1),1)
            predicted_price= self.estimator2l.predict(x1)[0] 
        else:
            #if test_row[(test_row.state.isnull()) & (test_row.life_sq.isnull())]:
            #print ((test_row.state.isnull().iloc[0]))
            if flag==1:
                x1 = test_row.full_sq
                x1 = x1.reshape(len(x1),1)
                
                #predicted_price= self.estimator2.predict(test_df)[0]
                predicted_price1= self.estimator2r.predict(test_row)[0] 
                predicted_price2= self.estimator2l.predict(x1)[0]
                predicted_price=0.3*predicted_price1+ 0.7*predicted_price2
                #if ((predicted_price > 6.8) & (dum<45)):
                #    predicted_price = predicted_price -4
                self.counts2_pred=self.counts2_pred
            else:
                test_row1=test_row1.drop(['sub_area', 'full_sq' ,'product_type'],axis=1)
                test_row1=test_row1.iloc[:,2:20]
                for fea in test_row1.columns:
                   test_row1[fea].fillna(test_row1[fea].median(), inplace=True)
                for fea in test_row1.columns:
                   test_row1[fea].fillna(0, inplace=True)
                pclass=self.TwoClassModel.predict(test_row1)
                ##print (pclass)
                #pclass=self.decision_clf.predict(test_row)[0]
                if pclass==-1:
                       predicted_price= 1
                else:  
                       
                        x1 = test_row.full_sq
                        x1 = x1.reshape(len(x1),1)
                
                        #predicted_price= self.estimator2.predict(test_df)[0]
                        predicted_price1= self.estimator2r.predict(test_row)[0] 
                        predicted_price2= self.estimator2l.predict(x1)[0]
                        predicted_price=0.3 * predicted_price1 + 0.7 * predicted_price2
                        #if ((predicted_price > 6.8) & (dum<45)):
                        #    predicted_price = predicted_price -4
                        self.counts2_pred=self.counts2_pred
                        #predicted_price= self.estimator2.predict(test_row)[0]
                       #if ((predicted_price > 6.8) & (dum<45)):
                       #      predicted_price = predicted_price 
#                if pclass==-1:
#                    if dum <39:
#                        predicted_price= 1
#                    else:
#                        predicted_price= 2
#                else:                        
#                        predicted_price= self.estimator3.predict(test_row)[0]
        #if test_row[2]< 35:
        #    predicted_price=predicted_price-1
        #print (predicted_price)
        #if predicted_price > 25:
        #    predicted_price=20
        return predicted_price

area='Poselenie Sosenskoe'

b1 = test_df[test_df.sub_area==area]
#print(b1)
########
########
#change method
a1=Poselenie_Sosenskoe(train_df[train_df.sub_area==area], TwoClassModel)
#b1=Poselenie_Sosenskoe.drop_test(b1)
#print (b1)
#Poselenie_Vnukovskoe.predict_price(s1)
predicted=[]
for pos, row in b1.iterrows():
    y=a1.predict_price(b1.loc[pos:pos+1])
    #print(len(b1.loc[pos:pos+1]))
    predicted.append(y)
print (predicted)
#plt.scatter(s1.full_sq,predicted, color='red')
grd=False
plt.figure(figsize=(7, 7))
plt.scatter(train_df[train_df.sub_area==area].full_sq, train_df[train_df.sub_area==area].price_doc)
#plt.show()
plt.xlim([0,200])
#plt.ylim([0,40])
plt.scatter(b1.full_sq,predicted, color='red')
plt.show()
