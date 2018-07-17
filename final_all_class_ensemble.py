# %load num-rooms.py
import pandas as pd
import gc
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
#class Predict_Class():
#    #_data = pd.DataFrame
#    def __init__(self, train_rows):
#        self.highcutoff = 100
#        self.lowpricecutoff = 2.1
#        sa=train_rows[train_rows.full_sq < self.highcutoff];
#        
#        s3=sa[sa.price_doc>self.lowpricecutoff]
#        s4=sa[sa.price_doc <= self.lowpricecutoff]
#
#        t3=s3.price_doc
#        t4=s4.price_doc
#        
#        s3=s3.drop(['price_doc','sub_area', 'full_sq' ,'product_type'],axis=1)
#        s3=s3.iloc[:,2:20]
#        for fea in s3.columns:
#                s3[fea].fillna(s3[fea].median(), inplace=True)
#        for fea in s3.columns:
#                s3[fea].fillna(0, inplace=True)
#        s4=s4.drop(['price_doc','sub_area', 'full_sq','product_type'],axis=1)
#        s4=s4.iloc[:,2:20]
#        for fea in s4.columns:
#                s4[fea].fillna(s4[fea].median(), inplace=True)
#        for fea in s4.columns:
#                s4[fea].fillna(0, inplace=True)
#        
#                
#        d3=np.ones(len(t3))
#        d4=np.ones(len(t4))*-1
#        print ("s3 shape",s3.shape)
#        X = pd.concat([s4,s4,s3,s4,s3,s4,s4,s4,s4])
#        y = np.concatenate([d4,d4,d3,d4,d3,d4,d4,d4,d4],axis=0)
#        self.y=y
#        self.X=X
#        #print (X)
#        #self.decision_clf = svm.SVC()
#        self.decision_clf = RandomForestClassifier()
#        self.decision_clf.fit(X, y)
#        
#        
#    def predict(self,test_row):
#                pclass=self.decision_clf.predict(test_row)[0]
#                return pclass
#
#
#selectedRows=(train_df.sub_area=='Kapotnja') | \
#             (train_df.sub_area=='Poselenie Sosenskoe') 
#
#TwoClassdata = train_df[selectedRows]
#
#TwoClassModel=Predict_Class(TwoClassdata)

class Predict_Class():
    #_data = pd.DataFrame
    def __init__(self, train_rows):
        self.highcutoff = 100
        
        sa=train_rows[train_rows.full_sq < self.highcutoff];
        
        s3=sa[sa.price_doc>3]
        s4=sa[sa.price_doc <= 2.1]

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
        X = pd.concat([s4,s3,s4])
        y = np.concatenate([d4,d3,d4],axis=0)
        self.y=y
        self.X=X
        #print (X)
        #self.decision_clf = svm.SVC()
        self.decision_clf = RandomForestClassifier()
        self.decision_clf.fit(X, y)
        
        
    def predict(self,test_row):
                pclass=self.decision_clf.predict(test_row)
                return pclass

selectedRows=(train_df.sub_area=='Chertanovo Severnoe') | \
             (train_df.sub_area=='Krjukovo') | \
             (train_df.sub_area=='Rostokino') | \
             (train_df.sub_area=='Basmannoe') |\
             (train_df.sub_area=='Poselenie Vnukovskoe') |\
             (train_df.sub_area=='Poselenie Desjonovskoe') |\
             (train_df.sub_area=='Poselenie Sosenskoe') |\
             (train_df.sub_area=='Poselenie Filimonkovskoe') |\
             (train_df.sub_area=='Poselenie Voskresenskoe') |\
             (train_df.sub_area=='Novo-Peredelkino') |\
             (train_df.sub_area=='Akademicheskoe') |\
             (train_df.sub_area=='Sokol') |\
             (train_df.sub_area=='Poselenie Krasnopahorskoe') |\
             (train_df.sub_area=='Zjuzino') |\
             (train_df.sub_area=='Matushkino') |\
             (train_df.sub_area=='Otradnoe') |\
             (train_df.sub_area=='Mitino') |\
             (train_df.sub_area=='Vojkovskoe') |\
             (train_df.sub_area=='Rjazanskij') |\
             (train_df.sub_area=='Severnoe Butovo') |\
             (train_df.sub_area=='Staroe Krjukovo') |\
             (train_df.sub_area=='Golovinskoe') |\
             (train_df.sub_area=='Kosino-Uhtomskoe') |\
             (train_df.sub_area=='Veshnjaki') |\
             (train_df.sub_area=='Horoshevo-Mnevniki') |\
             (train_df.sub_area=='Pechatniki') |\
             (train_df.sub_area=='Tekstil\'shhiki') |\
             (train_df.sub_area=='Solncevo') |\
             (train_df.sub_area=='Sviblovo') |\
             (train_df.sub_area=='Silino') |\
             (train_df.sub_area=='Butyrskoe') |\
             (train_df.sub_area=='Birjulevo Vostochnoe') |\
             (train_df.sub_area=='Caricyno') |\
             (train_df.sub_area=='Taganskoe') |\
             (train_df.sub_area=='Kapotnja') |\
             (train_df.sub_area=='Orehovo-Borisovo Juzhnoe') |\
             (train_df.sub_area=='Dmitrovskoe') |\
             (train_df.sub_area=='Juzhnoe Medvedkovo') |\
             (train_df.sub_area=='Sokolinaja Gora') |\
             (train_df.sub_area=='Lianozovo') |\
             (train_df.sub_area=='Zapadnoe Degunino') |\
             (train_df.sub_area=='Novogireevo') |\
             (train_df.sub_area=='Gol\'janovo') |\
             (train_df.sub_area=='Bogorodskoe') |\
             (train_df.sub_area=='Presnenskoe') |\
             (train_df.sub_area=='Timirjazevskoe') |\
             (train_df.sub_area=='Jasenevo') |\
             (train_df.sub_area=='Altuf\'evskoe') |\
             (train_df.sub_area=='Severnoe Medvedkovo') |\
             (train_df.sub_area=='Vyhino-Zhulebino') |\
             (train_df.sub_area=='Filevskij Park') |\
             (train_df.sub_area=='Kotlovka') |\
             (train_df.sub_area=='Jaroslavskoe') |\
             (train_df.sub_area=='Severnoe Izmajlovo') |\
             (train_df.sub_area=='Perovo') |\
             (train_df.sub_area=='Nizhegorodskoe') |\
             (train_df.sub_area=='Jakimanka') |\
             (train_df.sub_area=='Ivanovskoe') |\
             (train_df.sub_area=='Severnoe Tushino') |\
             (train_df.sub_area=='Nagatino-Sadovniki') |\
             (train_df.sub_area=='Ramenki') |\
             (train_df.sub_area=='Bibirevo') |\
             (train_df.sub_area=='Zjablikovo') |\
             (train_df.sub_area=='Meshhanskoe') |\
             (train_df.sub_area=='Chertanovo Juzhnoe') |\
             (train_df.sub_area=='Danilovskoe') |\
             (train_df.sub_area=='Hovrino') |\
             (train_df.sub_area=='Vostochnoe Degunino') |\
             (train_df.sub_area=='Birjulevo Zapadnoe') |\
             (train_df.sub_area=='Donskoe') |\
             (train_df.sub_area=='Chertanovo Central\'noe') |\
             (train_df.sub_area=='Losinoostrovskoe') |\
             (train_df.sub_area=='Vostochnoe Izmajlovo') |\
             (train_df.sub_area=='Orehovo-Borisovo Severnoe') |\
             (train_df.sub_area=='Preobrazhenskoe') |\
             (train_df.sub_area=='Fili Davydkovo') |\
             (train_df.sub_area=='Moskvorech\'e-Saburovo') |\
             (train_df.sub_area=='Teplyj Stan') |\
             (train_df.sub_area=='Lomonosovskoe') |\
             (train_df.sub_area=='Ljublino') |\
             (train_df.sub_area=='Strogino') |\
             (train_df.sub_area=='Koptevo') |\
             (train_df.sub_area=='Babushkinskoe') |\
             (train_df.sub_area=='Troparevo-Nikulino') |\
             (train_df.sub_area=='Cheremushki') |\
             (train_df.sub_area=='Levoberezhnoe') |\
             (train_df.sub_area=='Prospekt Vernadskogo') |\
             (train_df.sub_area=='Nagatinskij Zaton') |\
             (train_df.sub_area=='Savelki') |\
             (train_df.sub_area=='Poselenie Kokoshkino')


TwoClassdata = train_df[selectedRows]

TwoClassModel=Predict_Class(TwoClassdata)

class MasterModel_ALL():
    #_data = pd.DataFrame
    def __init__(self, train_rows, twoclass):
        
        #### Flags
        self.specialFlag=0 ### this is for special adjustment in predict
        self.FLAG1=1 ### this is 1 for combined s2 s3
        self.FLAG2=1 ### this is 1 for global TwoClass classifier
        
        
        
        self.highcutoff = 150
        self.highcutoff1=150
        self.lowpricecutoff = 2.2

        s1=train_rows[train_rows.full_sq > self.highcutoff];
        sa=train_rows[train_rows.full_sq < self.highcutoff];
        s2=sa[(sa.state.isnull()) & (sa.life_sq.isnull())]
        s3a = sa[~((sa.state.isnull()) & (sa.life_sq.isnull()))]
        s3=s3a[s3a.price_doc>self.lowpricecutoff]
        s4=s3a[s3a.price_doc <= self.lowpricecutoff]
        t1=s1.price_doc
        t2=s2.price_doc
        t3=s3.price_doc
        t4=s4.price_doc
        s1=s1.drop(['price_doc','sub_area','product_type'],axis=1)
        s1=s1.iloc[:,2:20]
        for fea in s1.columns:
                s1[fea].fillna(s1[fea].median(), inplace=True)
        for fea in s1.columns:
                s1[fea].fillna(0, inplace=True)
        s2=s2.drop(['price_doc','sub_area','product_type'],axis=1)
        s2=s2.iloc[:,2:20]
        for fea in s2.columns:
                s2[fea].fillna(s2[fea].median(), inplace=True)
        for fea in s2.columns:
                s2[fea].fillna(0, inplace=True)              
        s3=s3.drop(['price_doc','sub_area','product_type'],axis=1)
        s3=s3.iloc[:,2:20]
        for fea in s3.columns:
                s3[fea].fillna(s3[fea].median(), inplace=True)
        for fea in s3.columns:
                s3[fea].fillna(0, inplace=True)
        s4=s4.drop(['price_doc','sub_area','product_type'],axis=1)
        s4=s4.iloc[:,2:20]
        for fea in s4.columns:
                s4[fea].fillna(s4[fea].median(), inplace=True)
        for fea in s4.columns:
                s4[fea].fillna(0, inplace=True)
        
        sa=sa.drop(['price_doc','sub_area','product_type'],axis=1)
        sa=sa.iloc[:,2:20]
        for fea in sa.columns:
                sa[fea].fillna(sa[fea].median(), inplace=True)
        for fea in sa.columns:
                sa[fea].fillna(0, inplace=True)
                
                
        if self.FLAG1==1:
           X23 = pd.concat([s2,s3]) 
           y23 = np.concatenate([t2,t3],axis=0)
           random_estimator23=RandomForestRegressor(random_state=0, n_estimators=500)
           random_estimator23.fit(X23,y23)
           self.random_estimator2=random_estimator23
           self.random_estimator3=random_estimator23
           
           X23r=X23.full_sq
           X23r=X23r.reshape(len(X23r),1)
           y23r=y23.reshape(len(y23), 1)
           Linear_estimator23=LinearRegression()
           Linear_estimator23.fit(X23r, y23r)
           
           self.Linear_estimator2=Linear_estimator23
           self.Linear_estimator3=Linear_estimator23
           
        else:
            
            random_estimator2 = RandomForestRegressor(random_state=0, n_estimators=500)
            random_estimator2.fit(s2,t2)
            self.random_estimator2=random_estimator2
            
            Linear_estimator2 = LinearRegression()
            
            s2r=s2.full_sq
            s2r=s2r.reshape(len(s2r),1)
            t2r=t2
            t2r=t2r.reshape(len(t2r),1)
            Linear_estimator2.fit(s2r, t2r)
            self.Linear_estimator2=Linear_estimator2
            
            
            
            random_estimator3 = RandomForestRegressor(random_state=0, n_estimators=500)
            random_estimator3.fit(s3,t3)
            self.random_estimator3=random_estimator3
            
            Linear_estimator3 = LinearRegression()
            
            s3r=s3.full_sq
            s3r=s3r.reshape(len(s3r),1)
            t3r=t3
            t3r=t3r.reshape(len(t3r),1)
            Linear_estimator3.fit(s3r, t3r)
            self.Linear_estimator3=Linear_estimator3
            
            
            
           
        if self.FLAG2==1:
               self.TwoClassModel=twoclass
        else :
               
                    
                d3=np.ones(len(t3))
                d4=np.ones(len(t4))*-1
                          
                X = pd.concat([s3,s4])
                y = np.concatenate([d3,d4],axis=0)
                self.y=y
                self.X=X
                #print (X)
                #self.decision_clf = svm.SVC()
                TwoClassModel = RandomForestClassifier()
                TwoClassModel.fit(X, y)
                self.TwoClassModel=TwoClassModel
                
        
        random_estimator4 = RandomForestRegressor(random_state=0, n_estimators=500)
        random_estimator4.fit(s4,t4)
        self.random_estimator4=random_estimator4
        
        Linear_estimator4 = LinearRegression()        
        s4r=s4.full_sq
        s4r=s4r.reshape(len(s4r),1)
        t4r=t4
        t4r=t4r.reshape(len(t4r),1)
        Linear_estimator4.fit(s4r, t4r)
        self.Linear_estimator4=Linear_estimator4             

        
    def predict_price(self,test_row):
        testrow1=test_row
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
        if dum > self.highcutoff1:
            #predicted_price=5
            predicted_price1= self.random_estimator3.predict(test_row)[0] 
            
            x1 = test_row.full_sq
            x1 = x1.reshape(len(x1),1)
            
            predicted_price2= self.Linear_estimator3.predict(x1)[0]
            
        
            predicted_price=0.2*predicted_price1+ 0.8*predicted_price2
            
            
        else:
            if flag==1:
                
                predicted_price1= self.random_estimator2.predict(test_row)[0] 
                x1 = test_row.full_sq
                x1 = x1.reshape(len(x1),1)
                predicted_price2= self.Linear_estimator2.predict(x1)[0]
                predicted_price=0.2*predicted_price1+ 0.8*predicted_price2
            
            else:
                if self.FLAG2==1:
                    testrow1=testrow1.drop(['sub_area', 'full_sq' ,'product_type'],axis=1)
                    testrow1=testrow1.iloc[:,2:20]
                    for fea in testrow1.columns:
                              testrow1[fea].fillna(testrow1[fea].median(),inplace=True)
                    for fea in testrow1.columns:
                              testrow1[fea].fillna(0,inplace=True)
                              
                    pclass=self.TwoClassModel.predict(testrow1)[0]
                
                else:
                       pclass=self.TwoClassModel.predict(test_row)[0]
                       
                if pclass==-1:
                       #predicted_price= 1.5
                       predicted_price1= self.random_estimator4.predict(test_row)[0]
                       x1 = test_row.full_sq
                       x1 = x1.reshape(len(x1),1)
                       predicted_price2= self.Linear_estimator4.predict(x1)[0]
                       predicted_price=0.9*predicted_price1+ 0.1*predicted_price2
                else:                        
                       predicted_price1= self.random_estimator3.predict(test_row)[0] 
                       x1 = test_row.full_sq
                       x1 = x1.reshape(len(x1),1)
                       predicted_price2= self.Linear_estimator3.predict(x1)[0]
                       predicted_price=0.2*predicted_price1+ 0.8*predicted_price2
                       
        if self.specialFlag==1:
            if dum < 65 :
                 predicted_price=predicted_price-2
      
        return predicted_price

gc.collect()

class Poselenie_Vnukovskoe():
    #_data = pd.DataFrame
    def __init__(self, train_rows, twoclass):
        self.TwoClassModel=twoclass
        self.highcutoff = 250
        self.lowpricecutoff = 1.5
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
        estimator2 = RandomForestRegressor(random_state=0, n_estimators=500, verbose=1)
        #estimator2b = LinearRegression()
        #print (type(estimator))
        #s2r=s2.full_sq
        #s2r=s2r.reshape(len(s2r),1)
        #t2r=t2
        #t2r=t2r.reshape(len(t2r),1)
        estimator2.fit(s2, t2)
        self.estimator2=estimator2
        
    
        ########## ESTIMATOR 3 ##########
        estimator3 = RandomForestRegressor(random_state=0, n_estimators=500, verbose=1)
        #print (type(estimator))
        estimator3.fit(s3, t3)
        #estimator3.fit(sa, ta)
        self.estimator3=estimator3
        
        
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
        
          

        if dum > 250:
            predicted_price=5
        else:
            #if test_row[(test_row.state.isnull()) & (test_row.life_sq.isnull())]:
            #print ((test_row.state.isnull().iloc[0]))
            if flag==1:
                x1 = test_row.full_sq
                x1 = x1.reshape(len(x1),1)
                
                #predicted_price= self.estimator2.predict(test_df)[0]
                predicted_price= self.estimator2.predict(test_row)[0]
                if ((predicted_price > 6.8) & (dum<45)):
                    predicted_price = predicted_price -4
                self.counts2_pred=self.counts2_pred
            else:
                test_row1=test_row1.drop(['sub_area', 'full_sq' ,'product_type'],axis=1)
                test_row1=test_row1.iloc[:,2:20]
                for fea in test_row1.columns:
                   test_row1[fea].fillna(test_row1[fea].median(), inplace=True)
                for fea in test_row1.columns:
                   test_row1[fea].fillna(0, inplace=True)
                pclass=self.TwoClassModel.predict(test_row1)
                #print (pclass)
                if pclass==-1:
                       predicted_price= 1
                else:                        
                       predicted_price= self.estimator3.predict(test_row)[0]
                       if ((predicted_price > 6.8) & (dum<45)):
                             predicted_price = predicted_price -4
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
        return predicted_price
#######################################################################################################

class Chertanovo_Severnoe():
    #_data = pd.DataFrame
    def __init__(self, train_rows,twoclass):
        self.highcutoff = 250
        self.lowpricecutoff = 2.5
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
        X = pd.concat([s3,s4])
        y = np.concatenate([d3,d4],axis=0)
        self.y=y
        self.X=X
        #print (X)
        #self.decision_clf = svm.SVC()
        self.decision_clf = RandomForestClassifier()
        self.decision_clf.fit(X, y)
        
        ########## ESTIMATOR 2 ##########
        estimator2 = RandomForestRegressor(random_state=0, n_estimators=500, verbose=1)
        #estimator2 = LinearRegression()
        #print (type(estimator))
        s2r=s2.full_sq
        s2r=s2r.reshape(len(s2r),1)
        t2r=t2
        t2r=t2r.reshape(len(t2r),1)
        estimator2.fit(s2r, t2r)
        self.estimator2=estimator2
        
        ########## ESTIMATOR 3 ##########
        estimator3 = RandomForestRegressor(random_state=0, n_estimators=500, verbose=1)
        #print (type(estimator))
        estimator3.fit(s3, t3)
        #estimator3.fit(sa, ta)
        self.estimator3=estimator3
        
        
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
        #print ('krishna')
        #print (len(test_row))
        #print (type(test_row))
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
        if dum > 250:
            predicted_price=5
        else:
            #if test_row[(test_row.state.isnull()) & (test_row.life_sq.isnull())]:
            #print ((test_row.state.isnull().iloc[0]))
            if flag==1:
                x1 = test_row.full_sq
                x1 = x1.reshape(len(x1),1)
                
                #predicted_price= self.estimator2.predict(test_df)[0]
                predicted_price= self.estimator2.predict(x1)[0]
                self.counts2_pred=self.counts2_pred+1
            else:
                pclass=self.decision_clf.predict(test_row)[0]
                print (pclass)
                if pclass==-1:
                       predicted_price= 1
                else:                        
                       predicted_price= self.estimator3.predict(test_row)[0]
        #if test_row[2]< 35:
        #    predicted_price=predicted_price-1
        #print (predicted_price)
        return predicted_price
gc.collect()
###############################################################
    
class Chertanovo_Severnoe():
    #_data = pd.DataFrame
    def __init__(self, train_rows, twoclass):
        self.TwoClassModel=twoclass
        self.highcutoff = 250
        self.lowpricecutoff = 2.5
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

        estimator2r.fit(s2, t2)
        self.estimator2r=estimator2r
        
        
        s2r=s2.full_sq
        s2r=s2r.reshape(len(s2r),1)
        t2r=t2
        t2r=t2r.reshape(len(t2r),1)   
        estimator2l.fit(s2r, t2r)
        self.estimator2l=estimator2l
    
        ########## ESTIMATOR 3 ##########
        #estimator3 = RandomForestRegressor(random_state=0, n_estimators=500)
        estimator3=LinearRegression()
    
        s3r=s3.full_sq
        s3r=s3r.reshape(len(s3r),1)
        t3r=t3
        t3r=t3r.reshape(len(t3r),1)   
        estimator3.fit(s3r, t3r)
        
        
        self.estimator3=estimator3
        
        #print (type(estimator))
        estimator3.fit(s3, t3)
        #estimator3.fit(sa, ta)
        self.estimator3=estimator3
        
        
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
        
          

        if dum > 250:
            predicted_price=5
        else:
            #if test_row[(test_row.state.isnull()) & (test_row.life_sq.isnull())]:
            #print ((test_row.state.isnull().iloc[0]))
            if flag==1:
                x1 = test_row.full_sq
                x1 = x1.reshape(len(x1),1)
                
                #predicted_price= self.estimator2.predict(test_df)[0]
                predicted_price1= self.estimator2r.predict(test_row)[0] 
                predicted_price2= self.estimator2l.predict(x1)[0]
                predicted_price=mean([predicted_price1 ,predicted_price2])
                #if ((predicted_price > 6.8) & (dum<45)):
                #    predicted_price = predicted_price -4
                self.counts2_pred=self.counts2_pred
            else:
                #test_row1=test_row1.drop(['sub_area', 'full_sq' ,'product_type'],axis=1)
                #test_row1=test_row1.iloc[:,2:20]
                #for fea in test_row1.columns:
                #   test_row1[fea].fillna(test_row1[fea].median(), inplace=True)
                #for fea in test_row1.columns:
                #   test_row1[fea].fillna(0, inplace=True)
                #pclass=self.TwoClassModel.predict(test_row1)
                ##print (pclass)
                pclass=self.decision_clf.predict(test_row)[0]
                if pclass==-1:
                       predicted_price= 1
                else:                        
                       predicted_price= self.estimator3.predict(test_row)[0]
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
        if predicted_price > 25:
            predicted_price=20
        return predicted_price
gc.collect()
###############################################################

######################################################################################################################
class Krjukovo():
    #_data = pd.DataFrame
    def __init__(self, train_rows,twoclass):
        self.highcutoff = 250
        self.lowpricecutoff = 2
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
        #estimator2 = RandomForestRegressor(random_state=0, n_estimators=500, verbose=1)
        estimator2 = LinearRegression()
        #print (type(estimator))
        s2r=s2.full_sq
        s2r=s2r.reshape(len(s2r),1)
        t2r=t2
        t2r=t2r.reshape(len(t2r),1)
        estimator2.fit(s2r, t2r)
        self.estimator2=estimator2
        
        ########## ESTIMATOR 3 ##########
        estimator3 = RandomForestRegressor(random_state=0, n_estimators=500, verbose=1)
        #print (type(estimator))
        estimator3.fit(s3, t3)
        #estimator3.fit(sa, ta)
        self.estimator3=estimator3
        
        
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
        
          

        if dum > 250:
            predicted_price=5
        else:
            #if test_row[(test_row.state.isnull()) & (test_row.life_sq.isnull())]:
            #print ((test_row.state.isnull().iloc[0]))
            if flag==1:
                x1 = test_row.full_sq
                x1 = x1.reshape(len(x1),1)
                
                #predicted_price= self.estimator2.predict(test_df)[0]
                predicted_price= self.estimator2.predict(x1)[0]
                self.counts2_pred=self.counts2_pred+1
            else:
                pclass=self.decision_clf.predict(test_row)[0]
                #print (pclass)
                if pclass==-1:
                       predicted_price= 1
                else:                        
                       predicted_price= self.estimator3.predict(test_row)[0]
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
        return predicted_price
gc.collect()
###############################################################

######################################################################################################################
class Rostokino():
    #_data = pd.DataFrame
    def __init__(self, train_rows, twoclass):
        self.TwoClassModel=twoclass
        self.highcutoff = 150
        self.lowpricecutoff = 3
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
            predicted_price= self.estimator2l.predict(x1)[0] + 10
        else:
            #if test_row[(test_row.state.isnull()) & (test_row.life_sq.isnull())]:
            #print ((test_row.state.isnull().iloc[0]))
            if flag==1:
                x1 = test_row.full_sq
                x1 = x1.reshape(len(x1),1)
                
                #predicted_price= self.estimator2.predict(test_df)[0]
                predicted_price1= self.estimator2r.predict(test_row)[0] 
                predicted_price2= self.estimator2l.predict(x1)[0]
                predicted_price=mean([predicted_price1 ,predicted_price2])
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
                        predicted_price= np.mean([predicted_price1 ,predicted_price2])
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
        if predicted_price > 25:
            predicted_price=20
        return predicted_price
gc.collect()
###############################################################

######################################################################################################################
class Basmannoe():
    #_data = pd.DataFrame
    def __init__(self, train_rows, twoclass):
        self.TwoClassModel=twoclass
        self.highcutoff = 7000
        self.lowpricecutoff = 3
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
                predicted_price=np.mean([predicted_price1 ,predicted_price2])
                #if ((predicted_price > 6.8) & (dum<45)):
                #    predicted_price = predicted_price -4
                self.counts2_pred=self.counts2_pred
            else:
                #test_row1=test_row1.drop(['sub_area', 'full_sq' ,'product_type'],axis=1)
                #test_row1=test_row1.iloc[:,2:20]
                #for fea in test_row1.columns:
                   #test_row1[fea].fillna(test_row1[fea].median(), inplace=True)
                #for fea in test_row1.columns:
                   #test_row1[fea].fillna(0, inplace=True)
                #pclass=self.TwoClassModel.predict(test_row1)
                ##print (pclass)
                pclass=self.decision_clf.predict(test_row)[0]
                if pclass==-1:
                       predicted_price= 1
                else:  
                       
                        x1 = test_row.full_sq
                        x1 = x1.reshape(len(x1),1)
                
                        #predicted_price= self.estimator2.predict(test_df)[0]
                        predicted_price1= self.estimator2r.predict(test_row)[0] 
                        predicted_price2= self.estimator2l.predict(x1)[0]
                        predicted_price= np.mean([predicted_price1 ,predicted_price2])
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

gc.collect()
###############################################################
################################################################################################################
class Poselenie_Desjonovskoe():
    #_data = pd.DataFrame
    def __init__(self, train_rows, twoclass):
        self.TwoClassModel=twoclass
        self.highcutoff = 7000
        self.lowpricecutoff = 3
        #self.train_target=train_rows.price_doc
        #train_rows=train_rows.drop(['price_doc','sub_area','product_type'],axis=1)
        #train_rows1=train_rows.iloc[:,2:20]
        #self.train=train_rows1
        s1=train_rows[train_rows.full_sq > self.highcutoff];
        sa=train_rows[train_rows.full_sq < self.highcutoff];
        s2=sa[(sa.state.isnull()) & (sa.life_sq.isnull())]
        s3a = sa[~((sa.state.isnull()) & (sa.life_sq.isnull()))]
        s3=s3a[s3a.price_doc>self.lowpricecutoff+ 1]
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
        #self.decision_clf = svm.SVC()
        self.decision_clf = RandomForestClassifier()
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
                predicted_price=np.mean([predicted_price1 ,predicted_price2])
                #if ((predicted_price > 6.8) & (dum<45)):
                #    predicted_price = predicted_price -4
                self.counts2_pred=self.counts2_pred
            else:
                #test_row1=test_row1.drop(['sub_area', 'full_sq' ,'product_type'],axis=1)
                #test_row1=test_row1.iloc[:,2:20]
                #for fea in test_row1.columns:
                   #test_row1[fea].fillna(test_row1[fea].median(), inplace=True)
                #for fea in test_row1.columns:
                   #test_row1[fea].fillna(0, inplace=True)
                #pclass=self.TwoClassModel.predict(test_row1)
                ##print (pclass)
                pclass=self.decision_clf.predict(test_row)[0]
                if pclass==-1:
                       predicted_price= 1
                else:  
                       
                        x1 = test_row.full_sq
                        x1 = x1.reshape(len(x1),1)
                
                        #predicted_price= self.estimator2.predict(test_df)[0]
                        predicted_price1= self.estimator2r.predict(test_row)[0] 
                        predicted_price2= self.estimator2l.predict(x1)[0]
                        predicted_price= np.mean([predicted_price1 ,predicted_price2])
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
        if predicted_price > 5:
           predicted_price=predicted_price-1
     
        return predicted_price
gc.collect()
###############################################################
################################################################################################################
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
###############################################################################################################
class Poselenie_Filimonkovskoe():
    #_data = pd.DataFrame
    def __init__(self, train_rows, twoclass):
        self.TwoClassModel=twoclass
        self.highcutoff = 250
        self.lowpricecutoff = 1.1
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

        estimator2r.fit(s2, t2)
        self.estimator2r=estimator2r
        
        
        s2r=s2.full_sq
        s2r=s2r.reshape(len(s2r),1)
        t2r=t2
        t2r=t2r.reshape(len(t2r),1)   
        estimator2l.fit(s2r, t2r)
        self.estimator2l=estimator2l
    
        ########## ESTIMATOR 3 ##########
        #estimator3 = RandomForestRegressor(random_state=0, n_estimators=500)
        estimator3=LinearRegression()
   
        s3r=s3.full_sq
        s3r=s3r.reshape(len(s3r),1)
        t3r=t3
        t3r=t3r.reshape(len(t3r),1)   
        estimator3.fit(s3r, t3r)
        
        
        self.estimator3=estimator3
        
        #print (type(estimator))
        estimator3.fit(s3, t3)
        #estimator3.fit(sa, ta)
        self.estimator3=estimator3
        
        
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
                predicted_price=0.5*predicted_price1+ 0.5*predicted_price2
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
                       #
                        #x1 = test_row.full_sq
                        #x1 = x1.reshape(len(x1),1)
                #
                        ##predicted_price= self.estimator2.predict(test_df)[0]
                        #predicted_price1= self.estimator2r.predict(test_row)[0] 
                        #predicted_price2= self.estimator2l.predict(x1)[0]
                        #predicted_price=0.3 * predicted_price1 + 0.7 * predicted_price2
                        ##if ((predicted_price > 6.8) & (dum<45)):
                        ##    predicted_price = predicted_price -4
                        #self.counts2_pred=self.counts2_pred
                        predicted_price= self.estimator3.predict(test_row)[0]
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
gc.collect()
###############################################################
################################################################################################################
class Poselenie_Voskresenskoe():
    #_data = pd.DataFrame
    def __init__(self, train_rows, twoclass):
        
        #### Flags
        self.specialFlag=0 ### this is for special adjustment in predict
        self.FLAG1=1 ### this is 1 for combined s2 s3
        self.FLAG2=0 ### this is 1 for global TwoClass classifier
        
        
        
        self.highcutoff = 150
        self.highcutoff1=150
        self.lowpricecutoff = 2.2

        s1=train_rows[train_rows.full_sq > self.highcutoff];
        sa=train_rows[train_rows.full_sq < self.highcutoff];
        s2=sa[(sa.state.isnull()) & (sa.life_sq.isnull())]
        s3a = sa[~((sa.state.isnull()) & (sa.life_sq.isnull()))]
        s3=s3a[s3a.price_doc>self.lowpricecutoff]
        s4=s3a[s3a.price_doc <= self.lowpricecutoff]
        t1=s1.price_doc
        t2=s2.price_doc
        t3=s3.price_doc
        t4=s4.price_doc
        s1=s1.drop(['price_doc','sub_area','product_type'],axis=1)
        s1=s1.iloc[:,2:20]
        for fea in s1.columns:
                s1[fea].fillna(s1[fea].median(), inplace=True)
        for fea in s1.columns:
                s1[fea].fillna(0, inplace=True)
        s2=s2.drop(['price_doc','sub_area','product_type'],axis=1)
        s2=s2.iloc[:,2:20]
        for fea in s2.columns:
                s2[fea].fillna(s2[fea].median(), inplace=True)
        for fea in s2.columns:
                s2[fea].fillna(0, inplace=True)              
        s3=s3.drop(['price_doc','sub_area','product_type'],axis=1)
        s3=s3.iloc[:,2:20]
        for fea in s3.columns:
                s3[fea].fillna(s3[fea].median(), inplace=True)
        for fea in s3.columns:
                s3[fea].fillna(0, inplace=True)
        s4=s4.drop(['price_doc','sub_area','product_type'],axis=1)
        s4=s4.iloc[:,2:20]
        for fea in s4.columns:
                s4[fea].fillna(s4[fea].median(), inplace=True)
        for fea in s4.columns:
                s4[fea].fillna(0, inplace=True)
        
        sa=sa.drop(['price_doc','sub_area','product_type'],axis=1)
        sa=sa.iloc[:,2:20]
        for fea in sa.columns:
                sa[fea].fillna(sa[fea].median(), inplace=True)
        for fea in sa.columns:
                sa[fea].fillna(0, inplace=True)
                
                
        if self.FLAG1==1:
           X23 = pd.concat([s2,s3]) 
           y23 = np.concatenate([t2,t3],axis=0)
           random_estimator23=RandomForestRegressor(random_state=0, n_estimators=500)
           random_estimator23.fit(X23,y23)
           self.random_estimator2=random_estimator23
           self.random_estimator3=random_estimator23
           
           X23r=X23.full_sq
           X23r=X23r.reshape(len(X23r),1)
           y23r=y23.reshape(len(y23), 1)
           Linear_estimator23=LinearRegression()
           Linear_estimator23.fit(X23r, y23r)
           
           self.Linear_estimator2=Linear_estimator23
           self.Linear_estimator3=Linear_estimator23
           
        else:
            
            random_estimator2 = RandomForestRegressor(random_state=0, n_estimators=500)
            random_estimator2.fit(s2,t2)
            self.random_estimator2=random_estimator2
            
            Linear_estimator2 = LinearRegression()
            
            s2r=s2.full_sq
            s2r=s2r.reshape(len(s2r),1)
            t2r=t2
            t2r=t2r.reshape(len(t2r),1)
            Linear_estimator2.fit(s2r, t2r)
            self.Linear_estimator2=Linear_estimator2
            
            
            
            random_estimator3 = RandomForestRegressor(random_state=0, n_estimators=500)
            random_estimator3.fit(s3,t3)
            self.random_estimator3=random_estimator3
            
            Linear_estimator3 = LinearRegression()
            
            s3r=s3.full_sq
            s3r=s3r.reshape(len(s3r),1)
            t3r=t3
            t3r=t3r.reshape(len(t3r),1)
            Linear_estimator3.fit(s3r, t3r)
            self.Linear_estimator3=Linear_estimator3
            
            
            
           
        if self.FLAG2==1:
               self.TwoClassModel=twoclass
        else :
               
                    
                d3=np.ones(len(t3))
                d4=np.ones(len(t4))*-1
                          
                X = pd.concat([s3,s4])
                y = np.concatenate([d3,d4],axis=0)
                self.y=y
                self.X=X
                #print (X)
                #self.decision_clf = svm.SVC()
                TwoClassModel = RandomForestClassifier()
                TwoClassModel.fit(X, y)
                self.TwoClassModel=TwoClassModel
                
            
        Linear_estimator4 = LinearRegression()        
        s4r=s4.full_sq
        s4r=s4r.reshape(len(s4r),1)
        t4r=t4
        t4r=t4r.reshape(len(t4r),1)
        Linear_estimator4.fit(s4r, t4r)
        self.Linear_estimator4=Linear_estimator4             

        
    def predict_price(self,test_row):
  
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
        if dum > self.highcutoff1:
            #predicted_price=5
            predicted_price1= self.random_estimator3.predict(test_row)[0] 
            
            x1 = test_row.full_sq
            x1 = x1.reshape(len(x1),1)
            
            predicted_price2= self.Linear_estimator3.predict(x1)[0]
            
        
            predicted_price=0.2*predicted_price1+ 0.8*predicted_price2
            
            
        else:
            if flag==1:
                
                predicted_price1= self.random_estimator2.predict(test_row)[0] 
                x1 = test_row.full_sq
                x1 = x1.reshape(len(x1),1)
                predicted_price2= self.Linear_estimator2.predict(x1)[0]
                predicted_price=0.4*predicted_price1+ 0.6*predicted_price2
            
            else:
                
                pclass=self.TwoClassModel.predict(test_row)[0]
                if pclass==-1:
                       #predicted_price= 1.5
                       x1 = test_row.full_sq
                       x1 = x1.reshape(len(x1),1)
                       predicted_price= self.Linear_estimator4.predict(x1)[0]
                else:                        
                       predicted_price1= self.random_estimator3.predict(test_row)[0] 
                       x1 = test_row.full_sq
                       x1 = x1.reshape(len(x1),1)
                       predicted_price2= self.Linear_estimator3.predict(x1)[0]
                       predicted_price=0.5*predicted_price1+ 0.5*predicted_price2
                       
        if self.specialFlag==1:
            if predicted_price > 6.3 :
                 predicted_price=predicted_price+3
            elif (predicted_price <=6.3) & (predicted_price >3):
                 predicted_price=predicted_price+2
      
        return predicted_price
################################################################################################################
class Novo_Peredelkino():
    #_data = pd.DataFrame
    def __init__(self, train_rows, twoclass):
        
        #### Flags
        self.specialFlag=0 ### this is for special adjustment in predict
        self.FLAG1=1 ### this is 1 for combined s2 s3
        self.FLAG2=0 ### this is 1 for global TwoClass classifier
        
        
        
        self.highcutoff = 150
        self.highcutoff1=150
        self.lowpricecutoff = 4.1

        s1=train_rows[train_rows.full_sq > self.highcutoff];
        sa=train_rows[train_rows.full_sq < self.highcutoff];
        s2=sa[(sa.state.isnull()) & (sa.life_sq.isnull())]
        s3a = sa[~((sa.state.isnull()) & (sa.life_sq.isnull()))]
        s3=s3a[s3a.price_doc>self.lowpricecutoff]
        s4=s3a[s3a.price_doc <= self.lowpricecutoff]
        t1=s1.price_doc
        t2=s2.price_doc
        t3=s3.price_doc
        t4=s4.price_doc
        s1=s1.drop(['price_doc','sub_area','product_type'],axis=1)
        s1=s1.iloc[:,2:20]
        for fea in s1.columns:
                s1[fea].fillna(s1[fea].median(), inplace=True)
        for fea in s1.columns:
                s1[fea].fillna(0, inplace=True)
        s2=s2.drop(['price_doc','sub_area','product_type'],axis=1)
        s2=s2.iloc[:,2:20]
        for fea in s2.columns:
                s2[fea].fillna(s2[fea].median(), inplace=True)
        for fea in s2.columns:
                s2[fea].fillna(0, inplace=True)              
        s3=s3.drop(['price_doc','sub_area','product_type'],axis=1)
        s3=s3.iloc[:,2:20]
        for fea in s3.columns:
                s3[fea].fillna(s3[fea].median(), inplace=True)
        for fea in s3.columns:
                s3[fea].fillna(0, inplace=True)
        s4=s4.drop(['price_doc','sub_area','product_type'],axis=1)
        s4=s4.iloc[:,2:20]
        for fea in s4.columns:
                s4[fea].fillna(s4[fea].median(), inplace=True)
        for fea in s4.columns:
                s4[fea].fillna(0, inplace=True)
        
        sa=sa.drop(['price_doc','sub_area','product_type'],axis=1)
        sa=sa.iloc[:,2:20]
        for fea in sa.columns:
                sa[fea].fillna(sa[fea].median(), inplace=True)
        for fea in sa.columns:
                sa[fea].fillna(0, inplace=True)
                
                
        if self.FLAG1==1:
           X23 = pd.concat([s2,s3]) 
           y23 = np.concatenate([t2,t3],axis=0)
           random_estimator23=RandomForestRegressor(random_state=0, n_estimators=500)
           random_estimator23.fit(X23,y23)
           self.random_estimator2=random_estimator23
           self.random_estimator3=random_estimator23
           
           X23r=X23.full_sq
           X23r=X23r.reshape(len(X23r),1)
           y23r=y23.reshape(len(y23), 1)
           Linear_estimator23=LinearRegression()
           Linear_estimator23.fit(X23r, y23r)
           
           self.Linear_estimator2=Linear_estimator23
           self.Linear_estimator3=Linear_estimator23
           
        else:
            
            random_estimator2 = RandomForestRegressor(random_state=0, n_estimators=500)
            random_estimator2.fit(s2,t2)
            self.random_estimator2=random_estimator2
            
            Linear_estimator2 = LinearRegression()
            
            s2r=s2.full_sq
            s2r=s2r.reshape(len(s2r),1)
            t2r=t2
            t2r=t2r.reshape(len(t2r),1)
            Linear_estimator2.fit(s2r, t2r)
            self.Linear_estimator2=Linear_estimator2
            
            
            
            random_estimator3 = RandomForestRegressor(random_state=0, n_estimators=500)
            random_estimator3.fit(s3,t3)
            self.random_estimator3=random_estimator3
            
            Linear_estimator3 = LinearRegression()
            
            s3r=s3.full_sq
            s3r=s3r.reshape(len(s3r),1)
            t3r=t3
            t3r=t3r.reshape(len(t3r),1)
            Linear_estimator3.fit(s3r, t3r)
            self.Linear_estimator3=Linear_estimator3
            
            
            
           
        if self.FLAG2==1:
               self.TwoClassModel=twoclass
        else :
               
                    
                d3=np.ones(len(t3))
                d4=np.ones(len(t4))*-1
                          
                X = pd.concat([s3,s4])
                y = np.concatenate([d3,d4],axis=0)
                self.y=y
                self.X=X
                #print (X)
                #self.decision_clf = svm.SVC()
                TwoClassModel = RandomForestClassifier()
                TwoClassModel.fit(X, y)
                self.TwoClassModel=TwoClassModel
                
            
        Linear_estimator4 = LinearRegression()        
        s4r=s4.full_sq
        s4r=s4r.reshape(len(s4r),1)
        t4r=t4
        t4r=t4r.reshape(len(t4r),1)
        Linear_estimator4.fit(s4r, t4r)
        self.Linear_estimator4=Linear_estimator4             

        
    def predict_price(self,test_row):
        testrow1=test_row
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
        if dum > self.highcutoff1:
            #predicted_price=5
            predicted_price1= self.random_estimator3.predict(test_row)[0] 
            
            x1 = test_row.full_sq
            x1 = x1.reshape(len(x1),1)
            
            predicted_price2= self.Linear_estimator3.predict(x1)[0]
            
        
            predicted_price=0.2*predicted_price1+ 0.8*predicted_price2
            
            
        else:
            if flag==1:
                
                predicted_price1= self.random_estimator2.predict(test_row)[0] 
                x1 = test_row.full_sq
                x1 = x1.reshape(len(x1),1)
                predicted_price2= self.Linear_estimator2.predict(x1)[0]
                predicted_price=0.4*predicted_price1+ 0.6*predicted_price2
            
            else:
                if self.FLAG2==1:
                    testrow1=testrow1.drop(['sub_area', 'full_sq' ,'product_type'],axis=1)
                    testrow1=testrow1.iloc[:,2:20]
                    for fea in testrow1.columns:
                              testrow1[fea].fillna(testrow1[fea].median(),inplace=True)
                    for fea in testrow1.columns:
                              testrow1[fea].fillna(0,inplace=True)
                              
                    pclass=self.TwoClassModel.predict(testrow1)[0]
                
                else:
                       pclass=self.TwoClassModel.predict(test_row)[0]
                       
                if pclass==-1:
                       #predicted_price= 1.5
                       x1 = test_row.full_sq
                       x1 = x1.reshape(len(x1),1)
                       predicted_price= self.Linear_estimator4.predict(x1)[0]
                else:                        
                       predicted_price1= self.random_estimator3.predict(test_row)[0] 
                       x1 = test_row.full_sq
                       x1 = x1.reshape(len(x1),1)
                       predicted_price2= self.Linear_estimator3.predict(x1)[0]
                       predicted_price=0.9*predicted_price1+ 0.1*predicted_price2
                       
        if self.specialFlag==1:
            if dum < 65 :
                 predicted_price=predicted_price-2
      
        return predicted_price
gc.collect()
###############################################################
################################################################################################################
class Poselenie_Moskovskij():
    #_data = pd.DataFrame
    def __init__(self, train_rows, twoclass):
        
        #### Flags
        self.specialFlag=1 ### this is for special adjustment in predict
        self.FLAG1=1 ### this is 1 for combined s2 s3
        self.FLAG2=0 ### this is 1 for global TwoClass classifier
        
        
        
        self.highcutoff = 150
        self.highcutoff1=150
        self.lowpricecutoff = 2.1

        s1=train_rows[train_rows.full_sq > self.highcutoff];
        sa=train_rows[train_rows.full_sq < self.highcutoff];
        s2=sa[(sa.state.isnull()) & (sa.life_sq.isnull())]
        s3a = sa[~((sa.state.isnull()) & (sa.life_sq.isnull()))]
        s3=s3a[s3a.price_doc>self.lowpricecutoff]
        s4=s3a[s3a.price_doc <= self.lowpricecutoff]
        t1=s1.price_doc
        t2=s2.price_doc
        t3=s3.price_doc
        t4=s4.price_doc
        s1=s1.drop(['price_doc','sub_area','product_type'],axis=1)
        s1=s1.iloc[:,2:20]
        for fea in s1.columns:
                s1[fea].fillna(s1[fea].median(), inplace=True)
        for fea in s1.columns:
                s1[fea].fillna(0, inplace=True)
        s2=s2.drop(['price_doc','sub_area','product_type'],axis=1)
        s2=s2.iloc[:,2:20]
        for fea in s2.columns:
                s2[fea].fillna(s2[fea].median(), inplace=True)
        for fea in s2.columns:
                s2[fea].fillna(0, inplace=True)              
        s3=s3.drop(['price_doc','sub_area','product_type'],axis=1)
        s3=s3.iloc[:,2:20]
        for fea in s3.columns:
                s3[fea].fillna(s3[fea].median(), inplace=True)
        for fea in s3.columns:
                s3[fea].fillna(0, inplace=True)
        s4=s4.drop(['price_doc','sub_area','product_type'],axis=1)
        s4=s4.iloc[:,2:20]
        for fea in s4.columns:
                s4[fea].fillna(s4[fea].median(), inplace=True)
        for fea in s4.columns:
                s4[fea].fillna(0, inplace=True)
        
        sa=sa.drop(['price_doc','sub_area','product_type'],axis=1)
        sa=sa.iloc[:,2:20]
        for fea in sa.columns:
                sa[fea].fillna(sa[fea].median(), inplace=True)
        for fea in sa.columns:
                sa[fea].fillna(0, inplace=True)
                
                
        if self.FLAG1==1:
           X23 = pd.concat([s2,s3]) 
           y23 = np.concatenate([t2,t3],axis=0)
           random_estimator23=RandomForestRegressor(random_state=0, n_estimators=500)
           random_estimator23.fit(X23,y23)
           self.random_estimator2=random_estimator23
           self.random_estimator3=random_estimator23
           
           X23r=X23.full_sq
           X23r=X23r.reshape(len(X23r),1)
           y23r=y23.reshape(len(y23), 1)
           Linear_estimator23=LinearRegression()
           Linear_estimator23.fit(X23r, y23r)
           
           self.Linear_estimator2=Linear_estimator23
           self.Linear_estimator3=Linear_estimator23
           
        else:
            
            random_estimator2 = RandomForestRegressor(random_state=0, n_estimators=500)
            random_estimator2.fit(s2,t2)
            self.random_estimator2=random_estimator2
            
            Linear_estimator2 = LinearRegression()
            
            s2r=s2.full_sq
            s2r=s2r.reshape(len(s2r),1)
            t2r=t2
            t2r=t2r.reshape(len(t2r),1)
            Linear_estimator2.fit(s2r, t2r)
            self.Linear_estimator2=Linear_estimator2
            
            
            
            random_estimator3 = RandomForestRegressor(random_state=0, n_estimators=500)
            random_estimator3.fit(s3,t3)
            self.random_estimator3=random_estimator3
            
            Linear_estimator3 = LinearRegression()
            
            s3r=s3.full_sq
            s3r=s3r.reshape(len(s3r),1)
            t3r=t3
            t3r=t3r.reshape(len(t3r),1)
            Linear_estimator3.fit(s3r, t3r)
            self.Linear_estimator3=Linear_estimator3
            
            
            
           
        if self.FLAG2==1:
               self.TwoClassModel=twoclass
        else :
               
                    
                d3=np.ones(len(t3))
                d4=np.ones(len(t4))*-1
                          
                X = pd.concat([s3,s4])
                y = np.concatenate([d3,d4],axis=0)
                self.y=y
                self.X=X
                #print (X)
                #self.decision_clf = svm.SVC()
                TwoClassModel = RandomForestClassifier()
                TwoClassModel.fit(X, y)
                self.TwoClassModel=TwoClassModel
                
            
        Linear_estimator4 = LinearRegression()        
        s4r=s4.full_sq
        s4r=s4r.reshape(len(s4r),1)
        t4r=t4
        t4r=t4r.reshape(len(t4r),1)
        Linear_estimator4.fit(s4r, t4r)
        self.Linear_estimator4=Linear_estimator4             

        
    def predict_price(self,test_row):
        testrow1=test_row
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
        if dum > self.highcutoff1:
            #predicted_price=5
            predicted_price1= self.random_estimator3.predict(test_row)[0] 
            
            x1 = test_row.full_sq
            x1 = x1.reshape(len(x1),1)
            
            predicted_price2= self.Linear_estimator3.predict(x1)[0]
            
        
            predicted_price=0.2*predicted_price1+ 0.8*predicted_price2
            
            
        else:
            if flag==1:
                
                predicted_price1= self.random_estimator2.predict(test_row)[0] 
                x1 = test_row.full_sq
                x1 = x1.reshape(len(x1),1)
                predicted_price2= self.Linear_estimator2.predict(x1)[0]
                predicted_price=0.5*predicted_price1+ 0.5*predicted_price2
            
            else:
                if self.FLAG2==1:
                    testrow1=testrow1.drop(['sub_area', 'full_sq' ,'product_type'],axis=1)
                    testrow1=testrow1.iloc[:,2:20]
                    for fea in testrow1.columns:
                              testrow1[fea].fillna(testrow1[fea].median(),inplace=True)
                    for fea in testrow1.columns:
                              testrow1[fea].fillna(0,inplace=True)
                              
                    pclass=self.TwoClassModel.predict(testrow1)[0]
                
                else:
                       pclass=self.TwoClassModel.predict(test_row)[0]
                       
                if pclass==-1:
                       #predicted_price= 1.5
                       x1 = test_row.full_sq
                       x1 = x1.reshape(len(x1),1)
                       predicted_price= self.Linear_estimator4.predict(x1)[0]
                else:                        
                       predicted_price1= self.random_estimator3.predict(test_row)[0] 
                       x1 = test_row.full_sq
                       x1 = x1.reshape(len(x1),1)
                       predicted_price2= self.Linear_estimator3.predict(x1)[0]
                       predicted_price=0.7*predicted_price1+ 0.3*predicted_price2
                       
        if self.specialFlag==1:
            if (dum >45) & (dum <90) & (predicted_price > 4):
                 predicted_price=predicted_price-1.7
      
        return predicted_price


################################################################################################################
class Poselenie_Shherbinka():
    #_data = pd.DataFrame
    def __init__(self, train_rows, twoclass):
        
        #### Flags
        self.specialFlag=0 ### this is for special adjustment in predict
        self.FLAG1=1 ### this is 1 for combined s2 s3
        self.FLAG2=1 ### this is 1 for global TwoClass classifier
        
        
        
        self.highcutoff = 150
        self.highcutoff1=150
        self.lowpricecutoff = 1.8

        s1=train_rows[train_rows.full_sq > self.highcutoff];
        sa=train_rows[train_rows.full_sq < self.highcutoff];
        s2=sa[(sa.state.isnull()) & (sa.life_sq.isnull())]
        s3a = sa[~((sa.state.isnull()) & (sa.life_sq.isnull()))]
        s3=s3a[s3a.price_doc>self.lowpricecutoff]
        s4=s3a[s3a.price_doc <= self.lowpricecutoff]
        t1=s1.price_doc
        t2=s2.price_doc
        t3=s3.price_doc
        t4=s4.price_doc
        s1=s1.drop(['price_doc','sub_area','product_type'],axis=1)
        s1=s1.iloc[:,2:20]
        for fea in s1.columns:
                s1[fea].fillna(s1[fea].median(), inplace=True)
        for fea in s1.columns:
                s1[fea].fillna(0, inplace=True)
        s2=s2.drop(['price_doc','sub_area','product_type'],axis=1)
        s2=s2.iloc[:,2:20]
        for fea in s2.columns:
                s2[fea].fillna(s2[fea].median(), inplace=True)
        for fea in s2.columns:
                s2[fea].fillna(0, inplace=True)              
        s3=s3.drop(['price_doc','sub_area','product_type'],axis=1)
        s3=s3.iloc[:,2:20]
        for fea in s3.columns:
                s3[fea].fillna(s3[fea].median(), inplace=True)
        for fea in s3.columns:
                s3[fea].fillna(0, inplace=True)
        s4=s4.drop(['price_doc','sub_area','product_type'],axis=1)
        s4=s4.iloc[:,2:20]
        for fea in s4.columns:
                s4[fea].fillna(s4[fea].median(), inplace=True)
        for fea in s4.columns:
                s4[fea].fillna(0, inplace=True)
        
        sa=sa.drop(['price_doc','sub_area','product_type'],axis=1)
        sa=sa.iloc[:,2:20]
        for fea in sa.columns:
                sa[fea].fillna(sa[fea].median(), inplace=True)
        for fea in sa.columns:
                sa[fea].fillna(0, inplace=True)
                
                
        if self.FLAG1==1:
           X23 = pd.concat([s2,s3]) 
           y23 = np.concatenate([t2,t3],axis=0)
           random_estimator23=RandomForestRegressor(random_state=0, n_estimators=500)
           random_estimator23.fit(X23,y23)
           self.random_estimator2=random_estimator23
           self.random_estimator3=random_estimator23
           
           X23r=X23.full_sq
           X23r=X23r.reshape(len(X23r),1)
           y23r=y23.reshape(len(y23), 1)
           Linear_estimator23=LinearRegression()
           Linear_estimator23.fit(X23r, y23r)
           
           self.Linear_estimator2=Linear_estimator23
           self.Linear_estimator3=Linear_estimator23
           
        else:
            
            random_estimator2 = RandomForestRegressor(random_state=0, n_estimators=500)
            random_estimator2.fit(s2,t2)
            self.random_estimator2=random_estimator2
            
            Linear_estimator2 = LinearRegression()
            
            s2r=s2.full_sq
            s2r=s2r.reshape(len(s2r),1)
            t2r=t2
            t2r=t2r.reshape(len(t2r),1)
            Linear_estimator2.fit(s2r, t2r)
            self.Linear_estimator2=Linear_estimator2
            
            
            
            random_estimator3 = RandomForestRegressor(random_state=0, n_estimators=500)
            random_estimator3.fit(s3,t3)
            self.random_estimator3=random_estimator3
            
            Linear_estimator3 = LinearRegression()
            
            s3r=s3.full_sq
            s3r=s3r.reshape(len(s3r),1)
            t3r=t3
            t3r=t3r.reshape(len(t3r),1)
            Linear_estimator3.fit(s3r, t3r)
            self.Linear_estimator3=Linear_estimator3
            
            
            
           
        if self.FLAG2==1:
               self.TwoClassModel=twoclass
        else :
               
                    
                d3=np.ones(len(t3))
                d4=np.ones(len(t4))*-1
                          
                X = pd.concat([s3,s4])
                y = np.concatenate([d3,d4],axis=0)
                self.y=y
                self.X=X
                #print (X)
                #self.decision_clf = svm.SVC()
                TwoClassModel = RandomForestClassifier()
                TwoClassModel.fit(X, y)
                self.TwoClassModel=TwoClassModel
                
            
        Linear_estimator4 = LinearRegression()        
        s4r=s4.full_sq
        s4r=s4r.reshape(len(s4r),1)
        t4r=t4
        t4r=t4r.reshape(len(t4r),1)
        Linear_estimator4.fit(s4r, t4r)
        self.Linear_estimator4=Linear_estimator4             

        
    def predict_price(self,test_row):
        testrow1=test_row
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
        if dum > self.highcutoff1:
            #predicted_price=5
            predicted_price1= self.random_estimator3.predict(test_row)[0] 
            
            x1 = test_row.full_sq
            x1 = x1.reshape(len(x1),1)
            
            predicted_price2= self.Linear_estimator3.predict(x1)[0]
            
        
            predicted_price=0.2*predicted_price1+ 0.8*predicted_price2
            
            
        else:
            if flag==1:
                
                predicted_price1= self.random_estimator2.predict(test_row)[0] 
                x1 = test_row.full_sq
                x1 = x1.reshape(len(x1),1)
                predicted_price2= self.Linear_estimator2.predict(x1)[0]
                predicted_price=0.5*predicted_price1+ 0.5*predicted_price2
            
            else:
                if self.FLAG2==1:
                    testrow1=testrow1.drop(['sub_area', 'full_sq' ,'product_type'],axis=1)
                    testrow1=testrow1.iloc[:,2:20]
                    for fea in testrow1.columns:
                              testrow1[fea].fillna(testrow1[fea].median(),inplace=True)
                    for fea in testrow1.columns:
                              testrow1[fea].fillna(0,inplace=True)
                              
                    pclass=self.TwoClassModel.predict(testrow1)[0]
                
                else:
                       pclass=self.TwoClassModel.predict(test_row)[0]
                       
                if pclass==-1:
                       #predicted_price= 1.5
                       x1 = test_row.full_sq
                       x1 = x1.reshape(len(x1),1)
                       predicted_price= self.Linear_estimator4.predict(x1)[0]
                else:                        
                       predicted_price1= self.random_estimator3.predict(test_row)[0] 
                       x1 = test_row.full_sq
                       x1 = x1.reshape(len(x1),1)
                       predicted_price2= self.Linear_estimator3.predict(x1)[0]
                       predicted_price=1*predicted_price1+ 0*predicted_price2
                       
        if self.specialFlag==1:
            if (dum >45) & (dum <90) & (predicted_price > 4):
                 predicted_price=predicted_price-1.7
      
        return predicted_price
################################################################################################################
gc.collect()
###############################################################
####################################Poselenie Sosenskoe#######################
class Nekrasovka():
    #_data = pd.DataFrame
    def __init__(self, train_rows, twoclass):
        
        #### Flags
        self.specialFlag=1 ### this is for special adjustment in predict
        self.FLAG1=1 ### this is 1 for combined s2 s3
        self.FLAG2=0 ### this is 1 for global TwoClass classifier
        
        
        
        self.highcutoff = 150
        self.highcutoff1=150
        self.lowpricecutoff =3.1

        s1=train_rows[train_rows.full_sq > self.highcutoff];
        sa=train_rows[train_rows.full_sq < self.highcutoff];
        s2=sa[(sa.state.isnull()) & (sa.life_sq.isnull())]
        s3a = sa[~((sa.state.isnull()) & (sa.life_sq.isnull()))]
        s3=s3a[s3a.price_doc>self.lowpricecutoff]
        s4=s3a[s3a.price_doc <= self.lowpricecutoff]
        t1=s1.price_doc
        t2=s2.price_doc
        t3=s3.price_doc
        t4=s4.price_doc
        s1=s1.drop(['price_doc','sub_area','product_type'],axis=1)
        s1=s1.iloc[:,2:20]
        for fea in s1.columns:
                s1[fea].fillna(s1[fea].median(), inplace=True)
        for fea in s1.columns:
                s1[fea].fillna(0, inplace=True)
        s2=s2.drop(['price_doc','sub_area','product_type'],axis=1)
        s2=s2.iloc[:,2:20]
        for fea in s2.columns:
                s2[fea].fillna(s2[fea].median(), inplace=True)
        for fea in s2.columns:
                s2[fea].fillna(0, inplace=True)              
        s3=s3.drop(['price_doc','sub_area','product_type'],axis=1)
        s3=s3.iloc[:,2:20]
        for fea in s3.columns:
                s3[fea].fillna(s3[fea].median(), inplace=True)
        for fea in s3.columns:
                s3[fea].fillna(0, inplace=True)
        s4=s4.drop(['price_doc','sub_area','product_type'],axis=1)
        s4=s4.iloc[:,2:20]
        for fea in s4.columns:
                s4[fea].fillna(s4[fea].median(), inplace=True)
        for fea in s4.columns:
                s4[fea].fillna(0, inplace=True)
        
        sa=sa.drop(['price_doc','sub_area','product_type'],axis=1)
        sa=sa.iloc[:,2:20]
        for fea in sa.columns:
                sa[fea].fillna(sa[fea].median(), inplace=True)
        for fea in sa.columns:
                sa[fea].fillna(0, inplace=True)
                
                
        if self.FLAG1==1:
           X23 = pd.concat([s2,s3]) 
           y23 = np.concatenate([t2,t3],axis=0)
           random_estimator23=RandomForestRegressor(random_state=0, n_estimators=500)
           random_estimator23.fit(X23,y23)
           self.random_estimator2=random_estimator23
           self.random_estimator3=random_estimator23
           
           X23r=X23.full_sq
           X23r=X23r.reshape(len(X23r),1)
           y23r=y23.reshape(len(y23), 1)
           Linear_estimator23=LinearRegression()
           Linear_estimator23.fit(X23r, y23r)
           
           self.Linear_estimator2=Linear_estimator23
           self.Linear_estimator3=Linear_estimator23
           
        else:
            
            random_estimator2 = RandomForestRegressor(random_state=0, n_estimators=500)
            random_estimator2.fit(s2,t2)
            self.random_estimator2=random_estimator2
            
            Linear_estimator2 = LinearRegression()
            
            s2r=s2.full_sq
            s2r=s2r.reshape(len(s2r),1)
            t2r=t2
            t2r=t2r.reshape(len(t2r),1)
            Linear_estimator2.fit(s2r, t2r)
            self.Linear_estimator2=Linear_estimator2
            
            
            
            random_estimator3 = RandomForestRegressor(random_state=0, n_estimators=500)
            random_estimator3.fit(s3,t3)
            self.random_estimator3=random_estimator3
            
            Linear_estimator3 = LinearRegression()
            
            s3r=s3.full_sq
            s3r=s3r.reshape(len(s3r),1)
            t3r=t3
            t3r=t3r.reshape(len(t3r),1)
            Linear_estimator3.fit(s3r, t3r)
            self.Linear_estimator3=Linear_estimator3
            
            
            
           
        if self.FLAG2==1:
               self.TwoClassModel=twoclass
        else :
               
                    
                d3=np.ones(len(t3))
                d4=np.ones(len(t4))*-1
                          
                X = pd.concat([s3,s4])
                y = np.concatenate([d3,d4],axis=0)
                self.y=y
                self.X=X
                #print (X)
                #self.decision_clf = svm.SVC()
                TwoClassModel = RandomForestClassifier()
                TwoClassModel.fit(X, y)
                self.TwoClassModel=TwoClassModel
                
            
        Linear_estimator4 = LinearRegression()        
        s4r=s4.full_sq
        s4r=s4r.reshape(len(s4r),1)
        t4r=t4
        t4r=t4r.reshape(len(t4r),1)
        Linear_estimator4.fit(s4r, t4r)
        self.Linear_estimator4=Linear_estimator4             

        
    def predict_price(self,test_row):
        testrow1=test_row
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
        if dum > self.highcutoff1:
            #predicted_price=5
            predicted_price1= self.random_estimator3.predict(test_row)[0] 
            
            x1 = test_row.full_sq
            x1 = x1.reshape(len(x1),1)
            
            predicted_price2= self.Linear_estimator3.predict(x1)[0]
            
        
            predicted_price=0.5*predicted_price1+ 0.5*predicted_price2
            
            
        else:
            if flag==1:
                
                predicted_price1= self.random_estimator2.predict(test_row)[0] 
                x1 = test_row.full_sq
                x1 = x1.reshape(len(x1),1)
                predicted_price2= self.Linear_estimator2.predict(x1)[0]
                predicted_price=0.5*predicted_price1+ 0.5*predicted_price2
            
            else:
                if self.FLAG2==1:
                    testrow1=testrow1.drop(['sub_area', 'full_sq' ,'product_type'],axis=1)
                    testrow1=testrow1.iloc[:,2:20]
                    for fea in testrow1.columns:
                              testrow1[fea].fillna(testrow1[fea].median(),inplace=True)
                    for fea in testrow1.columns:
                              testrow1[fea].fillna(0,inplace=True)
                              
                    pclass=self.TwoClassModel.predict(testrow1)[0]
                
                else:
                       pclass=self.TwoClassModel.predict(test_row)[0]
                       
                if pclass==-1:
                       #predicted_price= 1.5
                       x1 = test_row.full_sq
                       x1 = x1.reshape(len(x1),1)
                       predicted_price= self.Linear_estimator4.predict(x1)[0]
                else:                        
                       predicted_price1= self.random_estimator3.predict(test_row)[0] 
                       x1 = test_row.full_sq
                       x1 = x1.reshape(len(x1),1)
                       predicted_price2= self.Linear_estimator3.predict(x1)[0]
                       predicted_price=0.7*predicted_price1+ 0.3*predicted_price2
                       
        if self.specialFlag==1:
            #if (dum >45) & (dum <90) & (predicted_price > 4):
            #     predicted_price=predicted_price-1.7
            if (dum >200) :
                 predicted_price=5.2      
        return predicted_price
gc.collect()
###############################################################
###############################################################################################################
class Akademicheskoe():
    #_data = pd.DataFrame
    def __init__(self, train_rows, twoclass):
        
        #### Flags
        self.specialFlag=1 ### this is for special adjustment in predict
        self.FLAG1=1 ### this is 1 for combined s2 s3
        self.FLAG2=0 ### this is 1 for global TwoClass classifier
        
        
        
        self.highcutoff = 150
        self.highcutoff1=150
        self.lowpricecutoff = 3.1

        s1=train_rows[train_rows.full_sq > self.highcutoff];
        sa=train_rows[train_rows.full_sq < self.highcutoff];
        s2=sa[(sa.state.isnull()) & (sa.life_sq.isnull())]
        s3a = sa[~((sa.state.isnull()) & (sa.life_sq.isnull()))]
        s3=s3a[s3a.price_doc>self.lowpricecutoff]
        s4=s3a[s3a.price_doc <= self.lowpricecutoff]
        t1=s1.price_doc
        t2=s2.price_doc
        t3=s3.price_doc
        t4=s4.price_doc
        s1=s1.drop(['price_doc','sub_area','product_type'],axis=1)
        s1=s1.iloc[:,2:20]
        for fea in s1.columns:
                s1[fea].fillna(s1[fea].median(), inplace=True)
        for fea in s1.columns:
                s1[fea].fillna(0, inplace=True)
        s2=s2.drop(['price_doc','sub_area','product_type'],axis=1)
        s2=s2.iloc[:,2:20]
        for fea in s2.columns:
                s2[fea].fillna(s2[fea].median(), inplace=True)
        for fea in s2.columns:
                s2[fea].fillna(0, inplace=True)              
        s3=s3.drop(['price_doc','sub_area','product_type'],axis=1)
        s3=s3.iloc[:,2:20]
        for fea in s3.columns:
                s3[fea].fillna(s3[fea].median(), inplace=True)
        for fea in s3.columns:
                s3[fea].fillna(0, inplace=True)
        s4=s4.drop(['price_doc','sub_area','product_type'],axis=1)
        s4=s4.iloc[:,2:20]
        for fea in s4.columns:
                s4[fea].fillna(s4[fea].median(), inplace=True)
        for fea in s4.columns:
                s4[fea].fillna(0, inplace=True)
        
        sa=sa.drop(['price_doc','sub_area','product_type'],axis=1)
        sa=sa.iloc[:,2:20]
        for fea in sa.columns:
                sa[fea].fillna(sa[fea].median(), inplace=True)
        for fea in sa.columns:
                sa[fea].fillna(0, inplace=True)
                
                
        if self.FLAG1==1:
           X23 = pd.concat([s2,s3]) 
           y23 = np.concatenate([t2,t3],axis=0)
           random_estimator23=RandomForestRegressor(random_state=0, n_estimators=500)
           random_estimator23.fit(X23,y23)
           self.random_estimator2=random_estimator23
           self.random_estimator3=random_estimator23
           
           X23r=X23.full_sq
           X23r=X23r.reshape(len(X23r),1)
           y23r=y23.reshape(len(y23), 1)
           Linear_estimator23=LinearRegression()
           Linear_estimator23.fit(X23r, y23r)
           
           self.Linear_estimator2=Linear_estimator23
           self.Linear_estimator3=Linear_estimator23
           
        else:
            
            random_estimator2 = RandomForestRegressor(random_state=0, n_estimators=500)
            random_estimator2.fit(s2,t2)
            self.random_estimator2=random_estimator2
            
            Linear_estimator2 = LinearRegression()
            
            s2r=s2.full_sq
            s2r=s2r.reshape(len(s2r),1)
            t2r=t2
            t2r=t2r.reshape(len(t2r),1)
            Linear_estimator2.fit(s2r, t2r)
            self.Linear_estimator2=Linear_estimator2
            
            
            
            random_estimator3 = RandomForestRegressor(random_state=0, n_estimators=500)
            random_estimator3.fit(s3,t3)
            self.random_estimator3=random_estimator3
            
            Linear_estimator3 = LinearRegression()
            
            s3r=s3.full_sq
            s3r=s3r.reshape(len(s3r),1)
            t3r=t3
            t3r=t3r.reshape(len(t3r),1)
            Linear_estimator3.fit(s3r, t3r)
            self.Linear_estimator3=Linear_estimator3
            
            
            
           
        if self.FLAG2==1:
               self.TwoClassModel=twoclass
        else :
               
                    
                d3=np.ones(len(t3))
                d4=np.ones(len(t4))*-1
                          
                X = pd.concat([s3,s4])
                y = np.concatenate([d3,d4],axis=0)
                self.y=y
                self.X=X
                #print (X)
                #self.decision_clf = svm.SVC()
                TwoClassModel = RandomForestClassifier()
                TwoClassModel.fit(X, y)
                self.TwoClassModel=TwoClassModel
                
            
        Linear_estimator4 = LinearRegression()        
        s4r=s4.full_sq
        s4r=s4r.reshape(len(s4r),1)
        t4r=t4
        t4r=t4r.reshape(len(t4r),1)
        Linear_estimator4.fit(s4r, t4r)
        self.Linear_estimator4=Linear_estimator4             

        
    def predict_price(self,test_row):
        testrow1=test_row
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
        if dum > self.highcutoff1:
            #predicted_price=5
            predicted_price1= self.random_estimator3.predict(test_row)[0] 
            
            x1 = test_row.full_sq
            x1 = x1.reshape(len(x1),1)
            
            predicted_price2= self.Linear_estimator3.predict(x1)[0]
            
        
            predicted_price=0.2*predicted_price1+ 0.8*predicted_price2
            
            
        else:
            if flag==1:
                
                predicted_price1= self.random_estimator2.predict(test_row)[0] 
                x1 = test_row.full_sq
                x1 = x1.reshape(len(x1),1)
                predicted_price2= self.Linear_estimator2.predict(x1)[0]
                predicted_price=0.5*predicted_price1+ 0.5*predicted_price2
            
            else:
                if self.FLAG2==1:
                    testrow1=testrow1.drop(['sub_area', 'full_sq' ,'product_type'],axis=1)
                    testrow1=testrow1.iloc[:,2:20]
                    for fea in testrow1.columns:
                              testrow1[fea].fillna(testrow1[fea].median(),inplace=True)
                    for fea in testrow1.columns:
                              testrow1[fea].fillna(0,inplace=True)
                              
                    pclass=self.TwoClassModel.predict(testrow1)[0]
                
                else:
                       pclass=self.TwoClassModel.predict(test_row)[0]
                       
                if pclass==-1:
                       #predicted_price= 1.5
                       x1 = test_row.full_sq
                       x1 = x1.reshape(len(x1),1)
                       predicted_price= self.Linear_estimator4.predict(x1)[0]
                else:                        
                       predicted_price1= self.random_estimator3.predict(test_row)[0] 
                       x1 = test_row.full_sq
                       x1 = x1.reshape(len(x1),1)
                       predicted_price2= self.Linear_estimator3.predict(x1)[0]
                       predicted_price=.5*predicted_price1+ .5*predicted_price2
                       
        if self.specialFlag==1:
            #if (dum >45) & (dum <90) & (predicted_price > 4):
            #     predicted_price=predicted_price-1.7
            if (dum >100) :
                 predicted_price +=  5    
        return predicted_price
gc.collect()
###############################################################
################################################################################################################
class Sokol():
    #_data = pd.DataFrame
    def __init__(self, train_rows, twoclass):
        
        #### Flags
        self.specialFlag= 0### this is for special adjustment in predict
        self.FLAG1=1 ### this is 1 for combined s2 s3
        self.FLAG2=1 ### this is 1 for global TwoClass classifier
        
        
        
        self.highcutoff = 200
        self.highcutoff1=200
        self.lowpricecutoff =3

        s1=train_rows[train_rows.full_sq > self.highcutoff];
        sa=train_rows[train_rows.full_sq < self.highcutoff];
        s2=sa[(sa.state.isnull()) & (sa.life_sq.isnull())]
        s3a = sa[~((sa.state.isnull()) & (sa.life_sq.isnull()))]
        s3=s3a[s3a.price_doc>self.lowpricecutoff]
        s4=s3a[s3a.price_doc <= self.lowpricecutoff]
        t1=s1.price_doc
        t2=s2.price_doc
        t3=s3.price_doc
        t4=s4.price_doc
        s1=s1.drop(['price_doc','sub_area','product_type'],axis=1)
        s1=s1.iloc[:,2:20]
        for fea in s1.columns:
                s1[fea].fillna(s1[fea].median(), inplace=True)
        for fea in s1.columns:
                s1[fea].fillna(0, inplace=True)
        s2=s2.drop(['price_doc','sub_area','product_type'],axis=1)
        s2=s2.iloc[:,2:20]
        for fea in s2.columns:
                s2[fea].fillna(s2[fea].median(), inplace=True)
        for fea in s2.columns:
                s2[fea].fillna(0, inplace=True)              
        s3=s3.drop(['price_doc','sub_area','product_type'],axis=1)
        s3=s3.iloc[:,2:20]
        for fea in s3.columns:
                s3[fea].fillna(s3[fea].median(), inplace=True)
        for fea in s3.columns:
                s3[fea].fillna(0, inplace=True)
        s4=s4.drop(['price_doc','sub_area','product_type'],axis=1)
        s4=s4.iloc[:,2:20]
        for fea in s4.columns:
                s4[fea].fillna(s4[fea].median(), inplace=True)
        for fea in s4.columns:
                s4[fea].fillna(0, inplace=True)
        
        sa=sa.drop(['price_doc','sub_area','product_type'],axis=1)
        sa=sa.iloc[:,2:20]
        for fea in sa.columns:
                sa[fea].fillna(sa[fea].median(), inplace=True)
        for fea in sa.columns:
                sa[fea].fillna(0, inplace=True)
                
                
        if self.FLAG1==1:
           X23 = pd.concat([s2,s3]) 
           y23 = np.concatenate([t2,t3],axis=0)
           random_estimator23=RandomForestRegressor(random_state=0, n_estimators=500)
           random_estimator23.fit(X23,y23)
           self.random_estimator2=random_estimator23
           self.random_estimator3=random_estimator23
           
           X23r=X23.full_sq
           X23r=X23r.reshape(len(X23r),1)
           y23r=y23.reshape(len(y23), 1)
           Linear_estimator23=LinearRegression()
           Linear_estimator23.fit(X23r, y23r)
           
           self.Linear_estimator2=Linear_estimator23
           self.Linear_estimator3=Linear_estimator23
           
        else:
            
            random_estimator2 = RandomForestRegressor(random_state=0, n_estimators=500)
            random_estimator2.fit(s2,t2)
            self.random_estimator2=random_estimator2
            
            Linear_estimator2 = LinearRegression()
            
            s2r=s2.full_sq
            s2r=s2r.reshape(len(s2r),1)
            t2r=t2
            t2r=t2r.reshape(len(t2r),1)
            Linear_estimator2.fit(s2r, t2r)
            self.Linear_estimator2=Linear_estimator2
            
            
            
            random_estimator3 = RandomForestRegressor(random_state=0, n_estimators=500)
            random_estimator3.fit(s3,t3)
            self.random_estimator3=random_estimator3
            
            Linear_estimator3 = LinearRegression()
            
            s3r=s3.full_sq
            s3r=s3r.reshape(len(s3r),1)
            t3r=t3
            t3r=t3r.reshape(len(t3r),1)
            Linear_estimator3.fit(s3r, t3r)
            self.Linear_estimator3=Linear_estimator3
            
            
            
           
        if self.FLAG2==1:
               self.TwoClassModel=twoclass
        else :
               
                    
                d3=np.ones(len(t3))
                d4=np.ones(len(t4))*-1
                          
                X = pd.concat([s3,s4])
                y = np.concatenate([d3,d4],axis=0)
                self.y=y
                self.X=X
                #print (X)
                #self.decision_clf = svm.SVC()
                TwoClassModel = RandomForestClassifier()
                TwoClassModel.fit(X, y)
                self.TwoClassModel=TwoClassModel
                
            
        Linear_estimator4 = LinearRegression()        
        s4r=s4.full_sq
        s4r=s4r.reshape(len(s4r),1)
        t4r=t4
        t4r=t4r.reshape(len(t4r),1)
        Linear_estimator4.fit(s4r, t4r)
        self.Linear_estimator4=Linear_estimator4             

        
    def predict_price(self,test_row):
        testrow1=test_row
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
        if dum > self.highcutoff1:
            #predicted_price=5
            predicted_price1= self.random_estimator3.predict(test_row)[0] 
            
            x1 = test_row.full_sq
            x1 = x1.reshape(len(x1),1)
            
            predicted_price2= self.Linear_estimator3.predict(x1)[0]
            
        
            predicted_price=0.5*predicted_price1+ 0.5*predicted_price2
            
            
        else:
            if flag==1:
                
                predicted_price1= self.random_estimator2.predict(test_row)[0] 
                x1 = test_row.full_sq
                x1 = x1.reshape(len(x1),1)
                predicted_price2= self.Linear_estimator2.predict(x1)[0]
                predicted_price=0.5*predicted_price1+ 0.5*predicted_price2
            
            else:
                if self.FLAG2==1:
                    testrow1=testrow1.drop(['sub_area', 'full_sq' ,'product_type'],axis=1)
                    testrow1=testrow1.iloc[:,2:20]
                    for fea in testrow1.columns:
                              testrow1[fea].fillna(testrow1[fea].median(),inplace=True)
                    for fea in testrow1.columns:
                              testrow1[fea].fillna(0,inplace=True)
                              
                    pclass=self.TwoClassModel.predict(testrow1)[0]
                
                else:
                       pclass=self.TwoClassModel.predict(test_row)[0]
                       
                if pclass==-1:
                       #predicted_price= 1.5
                       x1 = test_row.full_sq
                       x1 = x1.reshape(len(x1),1)
                       predicted_price= self.Linear_estimator4.predict(x1)[0]
                else:                        
                       predicted_price1= self.random_estimator3.predict(test_row)[0] 
                       x1 = test_row.full_sq
                       x1 = x1.reshape(len(x1),1)
                       predicted_price2= self.Linear_estimator3.predict(x1)[0]
                       predicted_price=0.7*predicted_price1+ 0.3*predicted_price2
                       
        if self.specialFlag==1:
            #if (dum >45) & (dum <90) & (predicted_price > 4):
            #     predicted_price=predicted_price-1.7
            if (dum >200) :
                 predicted_price=10     
        return predicted_price
gc.collect()
###############################################################
################################################################################################################
class Kuzminki():
    #_data = pd.DataFrame
    def __init__(self, train_rows, twoclass):
        
        #### Flags
        self.specialFlag=0 ### this is for special adjustment in predict
        self.FLAG1=1 ### this is 1 for combined s2 s3
        self.FLAG2=1 ### this is 1 for global TwoClass classifier
        
        
        
        self.highcutoff = 150
        self.highcutoff1=150
        self.lowpricecutoff = 3.1

        s1=train_rows[train_rows.full_sq > self.highcutoff];
        sa=train_rows[train_rows.full_sq < self.highcutoff];
        s2=sa[(sa.state.isnull()) & (sa.life_sq.isnull())]
        s3a = sa[~((sa.state.isnull()) & (sa.life_sq.isnull()))]
        s3=s3a[s3a.price_doc>self.lowpricecutoff]
        s4=s3a[s3a.price_doc <= self.lowpricecutoff]
        t1=s1.price_doc
        t2=s2.price_doc
        t3=s3.price_doc
        t4=s4.price_doc
        s1=s1.drop(['price_doc','sub_area','product_type'],axis=1)
        s1=s1.iloc[:,2:20]
        for fea in s1.columns:
                s1[fea].fillna(s1[fea].median(), inplace=True)
        for fea in s1.columns:
                s1[fea].fillna(0, inplace=True)
        s2=s2.drop(['price_doc','sub_area','product_type'],axis=1)
        s2=s2.iloc[:,2:20]
        for fea in s2.columns:
                s2[fea].fillna(s2[fea].median(), inplace=True)
        for fea in s2.columns:
                s2[fea].fillna(0, inplace=True)              
        s3=s3.drop(['price_doc','sub_area','product_type'],axis=1)
        s3=s3.iloc[:,2:20]
        for fea in s3.columns:
                s3[fea].fillna(s3[fea].median(), inplace=True)
        for fea in s3.columns:
                s3[fea].fillna(0, inplace=True)
        s4=s4.drop(['price_doc','sub_area','product_type'],axis=1)
        s4=s4.iloc[:,2:20]
        for fea in s4.columns:
                s4[fea].fillna(s4[fea].median(), inplace=True)
        for fea in s4.columns:
                s4[fea].fillna(0, inplace=True)
        
        sa=sa.drop(['price_doc','sub_area','product_type'],axis=1)
        sa=sa.iloc[:,2:20]
        for fea in sa.columns:
                sa[fea].fillna(sa[fea].median(), inplace=True)
        for fea in sa.columns:
                sa[fea].fillna(0, inplace=True)
                
                
        if self.FLAG1==1:
           X23 = pd.concat([s2,s3]) 
           y23 = np.concatenate([t2,t3],axis=0)
           random_estimator23=RandomForestRegressor(random_state=0, n_estimators=500)
           random_estimator23.fit(X23,y23)
           self.random_estimator2=random_estimator23
           self.random_estimator3=random_estimator23
           
           X23r=X23.full_sq
           X23r=X23r.reshape(len(X23r),1)
           y23r=y23.reshape(len(y23), 1)
           Linear_estimator23=LinearRegression()
           Linear_estimator23.fit(X23r, y23r)
           
           self.Linear_estimator2=Linear_estimator23
           self.Linear_estimator3=Linear_estimator23
           
        else:
            
            random_estimator2 = RandomForestRegressor(random_state=0, n_estimators=500)
            random_estimator2.fit(s2,t2)
            self.random_estimator2=random_estimator2
            
            Linear_estimator2 = LinearRegression()
            
            s2r=s2.full_sq
            s2r=s2r.reshape(len(s2r),1)
            t2r=t2
            t2r=t2r.reshape(len(t2r),1)
            Linear_estimator2.fit(s2r, t2r)
            self.Linear_estimator2=Linear_estimator2
            
            
            
            random_estimator3 = RandomForestRegressor(random_state=0, n_estimators=500)
            random_estimator3.fit(s3,t3)
            self.random_estimator3=random_estimator3
            
            Linear_estimator3 = LinearRegression()
            
            s3r=s3.full_sq
            s3r=s3r.reshape(len(s3r),1)
            t3r=t3
            t3r=t3r.reshape(len(t3r),1)
            Linear_estimator3.fit(s3r, t3r)
            self.Linear_estimator3=Linear_estimator3
            
            
            
           
        if self.FLAG2==1:
               self.TwoClassModel=twoclass
        else :
               
                    
                d3=np.ones(len(t3))
                d4=np.ones(len(t4))*-1
                          
                X = pd.concat([s3,s4])
                y = np.concatenate([d3,d4],axis=0)
                self.y=y
                self.X=X
                #print (X)
                #self.decision_clf = svm.SVC()
                TwoClassModel = RandomForestClassifier()
                TwoClassModel.fit(X, y)
                self.TwoClassModel=TwoClassModel
                
            
        Linear_estimator4 = LinearRegression()        
        s4r=s4.full_sq
        s4r=s4r.reshape(len(s4r),1)
        t4r=t4
        t4r=t4r.reshape(len(t4r),1)
        Linear_estimator4.fit(s4r, t4r)
        self.Linear_estimator4=Linear_estimator4             

        
    def predict_price(self,test_row):
        testrow1=test_row
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
        if dum > self.highcutoff1:
            #predicted_price=5
            predicted_price1= self.random_estimator3.predict(test_row)[0] 
            
            x1 = test_row.full_sq
            x1 = x1.reshape(len(x1),1)
            
            predicted_price2= self.Linear_estimator3.predict(x1)[0]
            
        
            predicted_price=0.2*predicted_price1+ 0.8*predicted_price2
            
            
        else:
            if flag==1:
                
                predicted_price1= self.random_estimator2.predict(test_row)[0] 
                x1 = test_row.full_sq
                x1 = x1.reshape(len(x1),1)
                predicted_price2= self.Linear_estimator2.predict(x1)[0]
                predicted_price=0.5*predicted_price1+ 0.5*predicted_price2
            
            else:
                if self.FLAG2==1:
                    testrow1=testrow1.drop(['sub_area', 'full_sq' ,'product_type'],axis=1)
                    testrow1=testrow1.iloc[:,2:20]
                    for fea in testrow1.columns:
                              testrow1[fea].fillna(testrow1[fea].median(),inplace=True)
                    for fea in testrow1.columns:
                              testrow1[fea].fillna(0,inplace=True)
                              
                    pclass=self.TwoClassModel.predict(testrow1)[0]
                
                else:
                       pclass=self.TwoClassModel.predict(test_row)[0]
                       
                if pclass==-1:
                       #predicted_price= 1.5
                       x1 = test_row.full_sq
                       x1 = x1.reshape(len(x1),1)
                       predicted_price= self.Linear_estimator4.predict(x1)[0]
                else:                        
                       predicted_price1= self.random_estimator3.predict(test_row)[0] 
                       x1 = test_row.full_sq
                       x1 = x1.reshape(len(x1),1)
                       predicted_price2= self.Linear_estimator3.predict(x1)[0]
                       predicted_price=.5*predicted_price1+ .5*predicted_price2
                       
        if self.specialFlag==1:
            #if (dum >45) & (dum <90) & (predicted_price > 4):
            #     predicted_price=predicted_price-1.7
            if (dum >100) :
                 predicted_price +=  5    
        return predicted_price


gc.collect()
###############################################################
################################################################################################################
class Zjuzino():
    #_data = pd.DataFrame
    def __init__(self, train_rows, twoclass):
        
        #### Flags
        self.specialFlag= 0### this is for special adjustment in predict
        self.FLAG1=1 ### this is 1 for combined s2 s3
        self.FLAG2=0 ### this is 1 for global TwoClass classifier
        
        
        
        self.highcutoff = 200
        self.highcutoff1=200
        self.lowpricecutoff =4

        s1=train_rows[train_rows.full_sq > self.highcutoff];
        sa=train_rows[train_rows.full_sq < self.highcutoff];
        s2=sa[(sa.state.isnull()) & (sa.life_sq.isnull())]
        s3a = sa[~((sa.state.isnull()) & (sa.life_sq.isnull()))]
        s3=s3a[s3a.price_doc>self.lowpricecutoff]
        s4=s3a[s3a.price_doc <= self.lowpricecutoff]
        t1=s1.price_doc
        t2=s2.price_doc
        t3=s3.price_doc
        t4=s4.price_doc
        s1=s1.drop(['price_doc','sub_area','product_type'],axis=1)
        s1=s1.iloc[:,2:20]
        for fea in s1.columns:
                s1[fea].fillna(s1[fea].median(), inplace=True)
        for fea in s1.columns:
                s1[fea].fillna(0, inplace=True)
        s2=s2.drop(['price_doc','sub_area','product_type'],axis=1)
        s2=s2.iloc[:,2:20]
        for fea in s2.columns:
                s2[fea].fillna(s2[fea].median(), inplace=True)
        for fea in s2.columns:
                s2[fea].fillna(0, inplace=True)              
        s3=s3.drop(['price_doc','sub_area','product_type'],axis=1)
        s3=s3.iloc[:,2:20]
        for fea in s3.columns:
                s3[fea].fillna(s3[fea].median(), inplace=True)
        for fea in s3.columns:
                s3[fea].fillna(0, inplace=True)
        s4=s4.drop(['price_doc','sub_area','product_type'],axis=1)
        s4=s4.iloc[:,2:20]
        for fea in s4.columns:
                s4[fea].fillna(s4[fea].median(), inplace=True)
        for fea in s4.columns:
                s4[fea].fillna(0, inplace=True)
        
        sa=sa.drop(['price_doc','sub_area','product_type'],axis=1)
        sa=sa.iloc[:,2:20]
        for fea in sa.columns:
                sa[fea].fillna(sa[fea].median(), inplace=True)
        for fea in sa.columns:
                sa[fea].fillna(0, inplace=True)
                
                
        if self.FLAG1==1:
           X23 = pd.concat([s2,s3]) 
           y23 = np.concatenate([t2,t3],axis=0)
           random_estimator23=RandomForestRegressor(random_state=0, n_estimators=500)
           random_estimator23.fit(X23,y23)
           self.random_estimator2=random_estimator23
           self.random_estimator3=random_estimator23
           
           X23r=X23.full_sq
           X23r=X23r.reshape(len(X23r),1)
           y23r=y23.reshape(len(y23), 1)
           Linear_estimator23=LinearRegression()
           Linear_estimator23.fit(X23r, y23r)
           
           self.Linear_estimator2=Linear_estimator23
           self.Linear_estimator3=Linear_estimator23
           
        else:
            
            random_estimator2 = RandomForestRegressor(random_state=0, n_estimators=500)
            random_estimator2.fit(s2,t2)
            self.random_estimator2=random_estimator2
            
            Linear_estimator2 = LinearRegression()
            
            s2r=s2.full_sq
            s2r=s2r.reshape(len(s2r),1)
            t2r=t2
            t2r=t2r.reshape(len(t2r),1)
            Linear_estimator2.fit(s2r, t2r)
            self.Linear_estimator2=Linear_estimator2
            
            
            
            random_estimator3 = RandomForestRegressor(random_state=0, n_estimators=500)
            random_estimator3.fit(s3,t3)
            self.random_estimator3=random_estimator3
            
            Linear_estimator3 = LinearRegression()
            
            s3r=s3.full_sq
            s3r=s3r.reshape(len(s3r),1)
            t3r=t3
            t3r=t3r.reshape(len(t3r),1)
            Linear_estimator3.fit(s3r, t3r)
            self.Linear_estimator3=Linear_estimator3
            
            
            
           
        if self.FLAG2==1:
               self.TwoClassModel=twoclass
        else :
               
                    
                d3=np.ones(len(t3))
                d4=np.ones(len(t4))*-1
                          
                X = pd.concat([s3,s4])
                y = np.concatenate([d3,d4],axis=0)
                self.y=y
                self.X=X
                #print (X)
                #self.decision_clf = svm.SVC()
                TwoClassModel = RandomForestClassifier()
                TwoClassModel.fit(X, y)
                self.TwoClassModel=TwoClassModel
                
            
        Linear_estimator4 = LinearRegression()        
        s4r=s4.full_sq
        s4r=s4r.reshape(len(s4r),1)
        t4r=t4
        t4r=t4r.reshape(len(t4r),1)
        Linear_estimator4.fit(s4r, t4r)
        self.Linear_estimator4=Linear_estimator4             

        
    def predict_price(self,test_row):
        testrow1=test_row
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
        if dum > self.highcutoff1:
            #predicted_price=5
            predicted_price1= self.random_estimator3.predict(test_row)[0] 
            
            x1 = test_row.full_sq
            x1 = x1.reshape(len(x1),1)
            
            predicted_price2= self.Linear_estimator3.predict(x1)[0]
            
        
            predicted_price=0.5*predicted_price1+ 0.5*predicted_price2
            
            
        else:
            if flag==1:
                
                predicted_price1= self.random_estimator2.predict(test_row)[0] 
                x1 = test_row.full_sq
                x1 = x1.reshape(len(x1),1)
                predicted_price2= self.Linear_estimator2.predict(x1)[0]
                predicted_price=0.5*predicted_price1+ 0.5*predicted_price2
            
            else:
                if self.FLAG2==1:
                    testrow1=testrow1.drop(['sub_area', 'full_sq' ,'product_type'],axis=1)
                    testrow1=testrow1.iloc[:,2:20]
                    for fea in testrow1.columns:
                              testrow1[fea].fillna(testrow1[fea].median(),inplace=True)
                    for fea in testrow1.columns:
                              testrow1[fea].fillna(0,inplace=True)
                              
                    pclass=self.TwoClassModel.predict(testrow1)[0]
                
                else:
                       pclass=self.TwoClassModel.predict(test_row)[0]
                       
                if pclass==-1:
                       #predicted_price= 1.5
                       x1 = test_row.full_sq
                       x1 = x1.reshape(len(x1),1)
                       predicted_price= self.Linear_estimator4.predict(x1)[0]
                else:                        
                       predicted_price1= self.random_estimator3.predict(test_row)[0] 
                       x1 = test_row.full_sq
                       x1 = x1.reshape(len(x1),1)
                       predicted_price2= self.Linear_estimator3.predict(x1)[0]
                       predicted_price=0.7*predicted_price1+ 0.3*predicted_price2
                       
        if self.specialFlag==1:
            #if (dum >45) & (dum <90) & (predicted_price > 4):
            #     predicted_price=predicted_price-1.7
            if (dum >200) :
                 predicted_price=10     
        return predicted_price
gc.collect()
###############################################################
################################################################################################################
class Poselenie_Novofedorovskoe():
    #_data = pd.DataFrame
    def __init__(self, train_rows, twoclass):
        
        #### Flags
        self.specialFlag=0 ### this is for special adjustment in predict
        self.FLAG1=1 ### this is 1 for combined s2 s3
        self.FLAG2=1 ### this is 1 for global TwoClass classifier
        
        
        
        self.highcutoff = 150
        self.highcutoff1=150
        self.lowpricecutoff =1.1

        s1=train_rows[train_rows.full_sq > self.highcutoff];
        sa=train_rows[train_rows.full_sq < self.highcutoff];
        s2=sa[(sa.state.isnull()) & (sa.life_sq.isnull())]
        s3a = sa[~((sa.state.isnull()) & (sa.life_sq.isnull()))]
        s3=s3a[s3a.price_doc>self.lowpricecutoff]
        s4=s3a[s3a.price_doc <= self.lowpricecutoff]
        t1=s1.price_doc
        t2=s2.price_doc
        t3=s3.price_doc
        t4=s4.price_doc
        s1=s1.drop(['price_doc','sub_area','product_type'],axis=1)
        s1=s1.iloc[:,2:20]
        for fea in s1.columns:
                s1[fea].fillna(s1[fea].median(), inplace=True)
        for fea in s1.columns:
                s1[fea].fillna(0, inplace=True)
        s2=s2.drop(['price_doc','sub_area','product_type'],axis=1)
        s2=s2.iloc[:,2:20]
        for fea in s2.columns:
                s2[fea].fillna(s2[fea].median(), inplace=True)
        for fea in s2.columns:
                s2[fea].fillna(0, inplace=True)              
        s3=s3.drop(['price_doc','sub_area','product_type'],axis=1)
        s3=s3.iloc[:,2:20]
        for fea in s3.columns:
                s3[fea].fillna(s3[fea].median(), inplace=True)
        for fea in s3.columns:
                s3[fea].fillna(0, inplace=True)
        s4=s4.drop(['price_doc','sub_area','product_type'],axis=1)
        s4=s4.iloc[:,2:20]
        for fea in s4.columns:
                s4[fea].fillna(s4[fea].median(), inplace=True)
        for fea in s4.columns:
                s4[fea].fillna(0, inplace=True)
        
        sa=sa.drop(['price_doc','sub_area','product_type'],axis=1)
        sa=sa.iloc[:,2:20]
        for fea in sa.columns:
                sa[fea].fillna(sa[fea].median(), inplace=True)
        for fea in sa.columns:
                sa[fea].fillna(0, inplace=True)
                
                
        if self.FLAG1==1:
           X23 = pd.concat([s2,s3]) 
           y23 = np.concatenate([t2,t3],axis=0)
           random_estimator23=RandomForestRegressor(random_state=0, n_estimators=500)
           random_estimator23.fit(X23,y23)
           self.random_estimator2=random_estimator23
           self.random_estimator3=random_estimator23
           
           X23r=X23.full_sq
           X23r=X23r.reshape(len(X23r),1)
           y23r=y23.reshape(len(y23), 1)
           Linear_estimator23=LinearRegression()
           Linear_estimator23.fit(X23r, y23r)
           
           self.Linear_estimator2=Linear_estimator23
           self.Linear_estimator3=Linear_estimator23
           
        else:
            
            random_estimator2 = RandomForestRegressor(random_state=0, n_estimators=500)
            random_estimator2.fit(s2,t2)
            self.random_estimator2=random_estimator2
            
            Linear_estimator2 = LinearRegression()
            
            s2r=s2.full_sq
            s2r=s2r.reshape(len(s2r),1)
            t2r=t2
            t2r=t2r.reshape(len(t2r),1)
            Linear_estimator2.fit(s2r, t2r)
            self.Linear_estimator2=Linear_estimator2
            
            
            
            random_estimator3 = RandomForestRegressor(random_state=0, n_estimators=500)
            random_estimator3.fit(s3,t3)
            self.random_estimator3=random_estimator3
            
            Linear_estimator3 = LinearRegression()
            
            s3r=s3.full_sq
            s3r=s3r.reshape(len(s3r),1)
            t3r=t3
            t3r=t3r.reshape(len(t3r),1)
            Linear_estimator3.fit(s3r, t3r)
            self.Linear_estimator3=Linear_estimator3
            
            
            
           
        if self.FLAG2==1:
               self.TwoClassModel=twoclass
        else :
               
                    
                d3=np.ones(len(t3))
                d4=np.ones(len(t4))*-1
                          
                X = pd.concat([s3,s4])
                y = np.concatenate([d3,d4],axis=0)
                self.y=y
                self.X=X
                #print (X)
                #self.decision_clf = svm.SVC()
                TwoClassModel = RandomForestClassifier()
                TwoClassModel.fit(X, y)
                self.TwoClassModel=TwoClassModel
                
            
        Linear_estimator4 = LinearRegression()        
        s4r=s4.full_sq
        s4r=s4r.reshape(len(s4r),1)
        t4r=t4
        t4r=t4r.reshape(len(t4r),1)
        Linear_estimator4.fit(s4r, t4r)
        self.Linear_estimator4=Linear_estimator4             

        
    def predict_price(self,test_row):
        testrow1=test_row
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
        if dum > self.highcutoff1:
            #predicted_price=5
            predicted_price1= self.random_estimator3.predict(test_row)[0] 
            
            x1 = test_row.full_sq
            x1 = x1.reshape(len(x1),1)
            
            predicted_price2= self.Linear_estimator3.predict(x1)[0]
            
        
            predicted_price=0.2*predicted_price1+ 0.8*predicted_price2
            
            
        else:
            if flag==1:
                
                predicted_price1= self.random_estimator2.predict(test_row)[0] 
                x1 = test_row.full_sq
                x1 = x1.reshape(len(x1),1)
                predicted_price2= self.Linear_estimator2.predict(x1)[0]
                predicted_price=0.5*predicted_price1+ 0.5*predicted_price2
            
            else:
                if self.FLAG2==1:
                    testrow1=testrow1.drop(['sub_area', 'full_sq' ,'product_type'],axis=1)
                    testrow1=testrow1.iloc[:,2:20]
                    for fea in testrow1.columns:
                              testrow1[fea].fillna(testrow1[fea].median(),inplace=True)
                    for fea in testrow1.columns:
                              testrow1[fea].fillna(0,inplace=True)
                              
                    pclass=self.TwoClassModel.predict(testrow1)[0]
                
                else:
                       pclass=self.TwoClassModel.predict(test_row)[0]
                       
                if pclass==-1:
                       #predicted_price= 1.5
                       x1 = test_row.full_sq
                       x1 = x1.reshape(len(x1),1)
                       predicted_price= self.Linear_estimator4.predict(x1)[0]
                else:                        
                       predicted_price1= self.random_estimator3.predict(test_row)[0] 
                       x1 = test_row.full_sq
                       x1 = x1.reshape(len(x1),1)
                       predicted_price2= self.Linear_estimator3.predict(x1)[0]
                       predicted_price=.5*predicted_price1+ .5*predicted_price2
                       
        if self.specialFlag==1:
            #if (dum >45) & (dum <90) & (predicted_price > 4):
            #     predicted_price=predicted_price-1.7
            if (dum >100) :
                 predicted_price +=  5    
        return predicted_price

################################################################################################################
class Juzhnoe_Butovo():
    #_data = pd.DataFrame
    def __init__(self, train_rows, twoclass):
        
        #### Flags
        self.specialFlag= 1### this is for special adjustment in predict
        self.FLAG1=0 ### this is 1 for combined s2 s3
        self.FLAG2=0 ### this is 1 for global TwoClass classifier
        
        
        
        self.highcutoff = 200
        self.highcutoff1=200
        self.lowpricecutoff =3.4

        s1=train_rows[train_rows.full_sq > self.highcutoff];
        sa=train_rows[train_rows.full_sq < self.highcutoff];
        s2=sa[(sa.state.isnull()) & (sa.life_sq.isnull())]
        s3a = sa[~((sa.state.isnull()) & (sa.life_sq.isnull()))]
        s3=s3a[s3a.price_doc>self.lowpricecutoff]
        s4=s3a[s3a.price_doc <= self.lowpricecutoff]
        t1=s1.price_doc
        t2=s2.price_doc
        t3=s3.price_doc
        t4=s4.price_doc
        s1=s1.drop(['price_doc','sub_area','product_type'],axis=1)
        s1=s1.iloc[:,2:20]
        for fea in s1.columns:
                s1[fea].fillna(s1[fea].median(), inplace=True)
        for fea in s1.columns:
                s1[fea].fillna(0, inplace=True)
        s2=s2.drop(['price_doc','sub_area','product_type'],axis=1)
        s2=s2.iloc[:,2:20]
        for fea in s2.columns:
                s2[fea].fillna(s2[fea].median(), inplace=True)
        for fea in s2.columns:
                s2[fea].fillna(0, inplace=True)              
        s3=s3.drop(['price_doc','sub_area','product_type'],axis=1)
        s3=s3.iloc[:,2:20]
        for fea in s3.columns:
                s3[fea].fillna(s3[fea].median(), inplace=True)
        for fea in s3.columns:
                s3[fea].fillna(0, inplace=True)
        s4=s4.drop(['price_doc','sub_area','product_type'],axis=1)
        s4=s4.iloc[:,2:20]
        for fea in s4.columns:
                s4[fea].fillna(s4[fea].median(), inplace=True)
        for fea in s4.columns:
                s4[fea].fillna(0, inplace=True)
        
        sa=sa.drop(['price_doc','sub_area','product_type'],axis=1)
        sa=sa.iloc[:,2:20]
        for fea in sa.columns:
                sa[fea].fillna(sa[fea].median(), inplace=True)
        for fea in sa.columns:
                sa[fea].fillna(0, inplace=True)
                
                
        if self.FLAG1==1:
           X23 = pd.concat([s2,s3]) 
           y23 = np.concatenate([t2,t3],axis=0)
           random_estimator23=RandomForestRegressor(random_state=0, n_estimators=500)
           random_estimator23.fit(X23,y23)
           self.random_estimator2=random_estimator23
           self.random_estimator3=random_estimator23
           
           X23r=X23.full_sq
           X23r=X23r.reshape(len(X23r),1)
           y23r=y23.reshape(len(y23), 1)
           Linear_estimator23=LinearRegression()
           Linear_estimator23.fit(X23r, y23r)
           
           self.Linear_estimator2=Linear_estimator23
           self.Linear_estimator3=Linear_estimator23
           
        else:
            
            random_estimator2 = RandomForestRegressor(random_state=0, n_estimators=500)
            random_estimator2.fit(s2,t2)
            self.random_estimator2=random_estimator2
            
            Linear_estimator2 = LinearRegression()
            
            s2r=s2.full_sq
            s2r=s2r.reshape(len(s2r),1)
            t2r=t2
            t2r=t2r.reshape(len(t2r),1)
            Linear_estimator2.fit(s2r, t2r)
            self.Linear_estimator2=Linear_estimator2
            
            
            
            random_estimator3 = RandomForestRegressor(random_state=0, n_estimators=500)
            random_estimator3.fit(s3,t3)
            self.random_estimator3=random_estimator3
            
            Linear_estimator3 = LinearRegression()
            
            s3r=s3.full_sq
            s3r=s3r.reshape(len(s3r),1)
            t3r=t3
            t3r=t3r.reshape(len(t3r),1)
            Linear_estimator3.fit(s3r, t3r)
            self.Linear_estimator3=Linear_estimator3
            
            
            
           
        if self.FLAG2==1:
               self.TwoClassModel=twoclass
        else :
               
                    
                d3=np.ones(len(t3))
                d4=np.ones(len(t4))*-1
                          
                X = pd.concat([s3,s4])
                y = np.concatenate([d3,d4],axis=0)
                self.y=y
                self.X=X
                #print (X)
                #self.decision_clf = svm.SVC()
                TwoClassModel = RandomForestClassifier()
                TwoClassModel.fit(X, y)
                self.TwoClassModel=TwoClassModel
                
            
        Linear_estimator4 = LinearRegression()        
        s4r=s4.full_sq
        s4r=s4r.reshape(len(s4r),1)
        t4r=t4
        t4r=t4r.reshape(len(t4r),1)
        Linear_estimator4.fit(s4r, t4r)
        self.Linear_estimator4=Linear_estimator4             

        
    def predict_price(self,test_row):
        testrow1=test_row
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
        if dum > self.highcutoff1:
            #predicted_price=5
            predicted_price1= self.random_estimator3.predict(test_row)[0] 
            
            x1 = test_row.full_sq
            x1 = x1.reshape(len(x1),1)
            
            predicted_price2= self.Linear_estimator3.predict(x1)[0]
            
        
            predicted_price=0.5*predicted_price1+ 0.5*predicted_price2
            
            
        else:
            if flag==1:
                
                predicted_price1= self.random_estimator2.predict(test_row)[0] 
                x1 = test_row.full_sq
                x1 = x1.reshape(len(x1),1)
                predicted_price2= self.Linear_estimator2.predict(x1)[0]
                predicted_price=0.5*predicted_price1+ 0.5*predicted_price2
            
            else:
                if self.FLAG2==1:
                    testrow1=testrow1.drop(['sub_area', 'full_sq' ,'product_type'],axis=1)
                    testrow1=testrow1.iloc[:,2:20]
                    for fea in testrow1.columns:
                              testrow1[fea].fillna(testrow1[fea].median(),inplace=True)
                    for fea in testrow1.columns:
                              testrow1[fea].fillna(0,inplace=True)
                              
                    pclass=self.TwoClassModel.predict(testrow1)[0]
                
                else:
                       pclass=self.TwoClassModel.predict(test_row)[0]
                       
                if pclass==-1:
                       #predicted_price= 1.5
                       x1 = test_row.full_sq
                       x1 = x1.reshape(len(x1),1)
                       predicted_price= self.Linear_estimator4.predict(x1)[0]
                else:                        
                       predicted_price1= self.random_estimator3.predict(test_row)[0] 
                       x1 = test_row.full_sq
                       x1 = x1.reshape(len(x1),1)
                       predicted_price2= self.Linear_estimator3.predict(x1)[0]
                       predicted_price=0.7*predicted_price1+ 0.3*predicted_price2
                       
        if self.specialFlag==1:
            #if (dum >45) & (dum <90) & (predicted_price > 4):
            #     predicted_price=predicted_price-1.7
            if (dum >150) :
                 predicted_price=15    
        return predicted_price

gc.collect()
###############################################################
################################################################################################################
class Matushkino():
    #_data = pd.DataFrame
    def __init__(self, train_rows, twoclass):
        
        #### Flags
        self.specialFlag=0 ### this is for special adjustment in predict
        self.FLAG1=1 ### this is 1 for combined s2 s3
        self.FLAG2=1 ### this is 1 for global TwoClass classifier
        
        
        
        self.highcutoff = 150
        self.highcutoff1=150
        self.lowpricecutoff =3

        s1=train_rows[train_rows.full_sq > self.highcutoff];
        sa=train_rows[train_rows.full_sq < self.highcutoff];
        s2=sa[(sa.state.isnull()) & (sa.life_sq.isnull())]
        s3a = sa[~((sa.state.isnull()) & (sa.life_sq.isnull()))]
        s3=s3a[s3a.price_doc>self.lowpricecutoff]
        s4=s3a[s3a.price_doc <= self.lowpricecutoff]
        t1=s1.price_doc
        t2=s2.price_doc
        t3=s3.price_doc
        t4=s4.price_doc
        s1=s1.drop(['price_doc','sub_area','product_type'],axis=1)
        s1=s1.iloc[:,2:20]
        for fea in s1.columns:
                s1[fea].fillna(s1[fea].median(), inplace=True)
        for fea in s1.columns:
                s1[fea].fillna(0, inplace=True)
        s2=s2.drop(['price_doc','sub_area','product_type'],axis=1)
        s2=s2.iloc[:,2:20]
        for fea in s2.columns:
                s2[fea].fillna(s2[fea].median(), inplace=True)
        for fea in s2.columns:
                s2[fea].fillna(0, inplace=True)              
        s3=s3.drop(['price_doc','sub_area','product_type'],axis=1)
        s3=s3.iloc[:,2:20]
        for fea in s3.columns:
                s3[fea].fillna(s3[fea].median(), inplace=True)
        for fea in s3.columns:
                s3[fea].fillna(0, inplace=True)
        s4=s4.drop(['price_doc','sub_area','product_type'],axis=1)
        s4=s4.iloc[:,2:20]
        for fea in s4.columns:
                s4[fea].fillna(s4[fea].median(), inplace=True)
        for fea in s4.columns:
                s4[fea].fillna(0, inplace=True)
        
        sa=sa.drop(['price_doc','sub_area','product_type'],axis=1)
        sa=sa.iloc[:,2:20]
        for fea in sa.columns:
                sa[fea].fillna(sa[fea].median(), inplace=True)
        for fea in sa.columns:
                sa[fea].fillna(0, inplace=True)
                
                
        if self.FLAG1==1:
           X23 = pd.concat([s2,s3]) 
           y23 = np.concatenate([t2,t3],axis=0)
           random_estimator23=RandomForestRegressor(random_state=0, n_estimators=500)
           random_estimator23.fit(X23,y23)
           self.random_estimator2=random_estimator23
           self.random_estimator3=random_estimator23
           
           X23r=X23.full_sq
           X23r=X23r.reshape(len(X23r),1)
           y23r=y23.reshape(len(y23), 1)
           Linear_estimator23=LinearRegression()
           Linear_estimator23.fit(X23r, y23r)
           
           self.Linear_estimator2=Linear_estimator23
           self.Linear_estimator3=Linear_estimator23
           
        else:
            
            random_estimator2 = RandomForestRegressor(random_state=0, n_estimators=500)
            random_estimator2.fit(s2,t2)
            self.random_estimator2=random_estimator2
            
            Linear_estimator2 = LinearRegression()
            
            s2r=s2.full_sq
            s2r=s2r.reshape(len(s2r),1)
            t2r=t2
            t2r=t2r.reshape(len(t2r),1)
            Linear_estimator2.fit(s2r, t2r)
            self.Linear_estimator2=Linear_estimator2
            
            
            
            random_estimator3 = RandomForestRegressor(random_state=0, n_estimators=500)
            random_estimator3.fit(s3,t3)
            self.random_estimator3=random_estimator3
            
            Linear_estimator3 = LinearRegression()
            
            s3r=s3.full_sq
            s3r=s3r.reshape(len(s3r),1)
            t3r=t3
            t3r=t3r.reshape(len(t3r),1)
            Linear_estimator3.fit(s3r, t3r)
            self.Linear_estimator3=Linear_estimator3
            
            
            
           
        if self.FLAG2==1:
               self.TwoClassModel=twoclass
        else :
               
                    
                d3=np.ones(len(t3))
                d4=np.ones(len(t4))*-1
                          
                X = pd.concat([s3,s4])
                y = np.concatenate([d3,d4],axis=0)
                self.y=y
                self.X=X
                #print (X)
                #self.decision_clf = svm.SVC()
                TwoClassModel = RandomForestClassifier()
                TwoClassModel.fit(X, y)
                self.TwoClassModel=TwoClassModel
                
            
        Linear_estimator4 = LinearRegression()        
        s4r=s4.full_sq
        s4r=s4r.reshape(len(s4r),1)
        t4r=t4
        t4r=t4r.reshape(len(t4r),1)
        Linear_estimator4.fit(s4r, t4r)
        self.Linear_estimator4=Linear_estimator4             

        
    def predict_price(self,test_row):
        testrow1=test_row
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
        if dum > self.highcutoff1:
            #predicted_price=5
            predicted_price1= self.random_estimator3.predict(test_row)[0] 
            
            x1 = test_row.full_sq
            x1 = x1.reshape(len(x1),1)
            
            predicted_price2= self.Linear_estimator3.predict(x1)[0]
            
        
            predicted_price=0.2*predicted_price1+ 0.8*predicted_price2
            
            
        else:
            if flag==1:
                
                predicted_price1= self.random_estimator2.predict(test_row)[0] 
                x1 = test_row.full_sq
                x1 = x1.reshape(len(x1),1)
                predicted_price2= self.Linear_estimator2.predict(x1)[0]
                predicted_price=0.5*predicted_price1+ 0.5*predicted_price2
            
            else:
                if self.FLAG2==1:
                    testrow1=testrow1.drop(['sub_area', 'full_sq' ,'product_type'],axis=1)
                    testrow1=testrow1.iloc[:,2:20]
                    for fea in testrow1.columns:
                              testrow1[fea].fillna(testrow1[fea].median(),inplace=True)
                    for fea in testrow1.columns:
                              testrow1[fea].fillna(0,inplace=True)
                              
                    pclass=self.TwoClassModel.predict(testrow1)[0]
                
                else:
                       pclass=self.TwoClassModel.predict(test_row)[0]
                       
                if pclass==-1:
                       #predicted_price= 1.5
                       x1 = test_row.full_sq
                       x1 = x1.reshape(len(x1),1)
                       predicted_price= self.Linear_estimator4.predict(x1)[0]
                else:                        
                       predicted_price1= self.random_estimator3.predict(test_row)[0] 
                       x1 = test_row.full_sq
                       x1 = x1.reshape(len(x1),1)
                       predicted_price2= self.Linear_estimator3.predict(x1)[0]
                       predicted_price=.5*predicted_price1+ .5*predicted_price2
                       
        if self.specialFlag==1:
            #if (dum >45) & (dum <90) & (predicted_price > 4):
            #     predicted_price=predicted_price-1.7
            if (dum >100) :
                 predicted_price +=  5    
        return predicted_price

gc.collect()
###############################################################
################################################################################################################
class Otradnoe():
    #_data = pd.DataFrame
    def __init__(self, train_rows, twoclass):
        
        #### Flags
        self.specialFlag=0 ### this is for special adjustment in predict
        self.FLAG1=1 ### this is 1 for combined s2 s3
        self.FLAG2=1 ### this is 1 for global TwoClass classifier
        
        
        
        self.highcutoff = 150
        self.highcutoff1=150
        self.lowpricecutoff =3

        s1=train_rows[train_rows.full_sq > self.highcutoff];
        sa=train_rows[train_rows.full_sq < self.highcutoff];
        s2=sa[(sa.state.isnull()) & (sa.life_sq.isnull())]
        s3a = sa[~((sa.state.isnull()) & (sa.life_sq.isnull()))]
        s3=s3a[s3a.price_doc>self.lowpricecutoff]
        s4=s3a[s3a.price_doc <= self.lowpricecutoff]
        t1=s1.price_doc
        t2=s2.price_doc
        t3=s3.price_doc
        t4=s4.price_doc
        s1=s1.drop(['price_doc','sub_area','product_type'],axis=1)
        s1=s1.iloc[:,2:20]
        for fea in s1.columns:
                s1[fea].fillna(s1[fea].median(), inplace=True)
        for fea in s1.columns:
                s1[fea].fillna(0, inplace=True)
        s2=s2.drop(['price_doc','sub_area','product_type'],axis=1)
        s2=s2.iloc[:,2:20]
        for fea in s2.columns:
                s2[fea].fillna(s2[fea].median(), inplace=True)
        for fea in s2.columns:
                s2[fea].fillna(0, inplace=True)              
        s3=s3.drop(['price_doc','sub_area','product_type'],axis=1)
        s3=s3.iloc[:,2:20]
        for fea in s3.columns:
                s3[fea].fillna(s3[fea].median(), inplace=True)
        for fea in s3.columns:
                s3[fea].fillna(0, inplace=True)
        s4=s4.drop(['price_doc','sub_area','product_type'],axis=1)
        s4=s4.iloc[:,2:20]
        for fea in s4.columns:
                s4[fea].fillna(s4[fea].median(), inplace=True)
        for fea in s4.columns:
                s4[fea].fillna(0, inplace=True)
        
        sa=sa.drop(['price_doc','sub_area','product_type'],axis=1)
        sa=sa.iloc[:,2:20]
        for fea in sa.columns:
                sa[fea].fillna(sa[fea].median(), inplace=True)
        for fea in sa.columns:
                sa[fea].fillna(0, inplace=True)
                
                
        if self.FLAG1==1:
           X23 = pd.concat([s2,s3]) 
           y23 = np.concatenate([t2,t3],axis=0)
           random_estimator23=RandomForestRegressor(random_state=0, n_estimators=500)
           random_estimator23.fit(X23,y23)
           self.random_estimator2=random_estimator23
           self.random_estimator3=random_estimator23
           
           X23r=X23.full_sq
           X23r=X23r.reshape(len(X23r),1)
           y23r=y23.reshape(len(y23), 1)
           Linear_estimator23=LinearRegression()
           Linear_estimator23.fit(X23r, y23r)
           
           self.Linear_estimator2=Linear_estimator23
           self.Linear_estimator3=Linear_estimator23
           
        else:
            
            random_estimator2 = RandomForestRegressor(random_state=0, n_estimators=500)
            random_estimator2.fit(s2,t2)
            self.random_estimator2=random_estimator2
            
            Linear_estimator2 = LinearRegression()
            
            s2r=s2.full_sq
            s2r=s2r.reshape(len(s2r),1)
            t2r=t2
            t2r=t2r.reshape(len(t2r),1)
            Linear_estimator2.fit(s2r, t2r)
            self.Linear_estimator2=Linear_estimator2
            
            
            
            random_estimator3 = RandomForestRegressor(random_state=0, n_estimators=500)
            random_estimator3.fit(s3,t3)
            self.random_estimator3=random_estimator3
            
            Linear_estimator3 = LinearRegression()
            
            s3r=s3.full_sq
            s3r=s3r.reshape(len(s3r),1)
            t3r=t3
            t3r=t3r.reshape(len(t3r),1)
            Linear_estimator3.fit(s3r, t3r)
            self.Linear_estimator3=Linear_estimator3
            
            
            
           
        if self.FLAG2==1:
               self.TwoClassModel=twoclass
        else :
               
                    
                d3=np.ones(len(t3))
                d4=np.ones(len(t4))*-1
                          
                X = pd.concat([s3,s4])
                y = np.concatenate([d3,d4],axis=0)
                self.y=y
                self.X=X
                #print (X)
                #self.decision_clf = svm.SVC()
                TwoClassModel = RandomForestClassifier()
                TwoClassModel.fit(X, y)
                self.TwoClassModel=TwoClassModel
                
            
        Linear_estimator4 = LinearRegression()        
        s4r=s4.full_sq
        s4r=s4r.reshape(len(s4r),1)
        t4r=t4
        t4r=t4r.reshape(len(t4r),1)
        Linear_estimator4.fit(s4r, t4r)
        self.Linear_estimator4=Linear_estimator4             

        
    def predict_price(self,test_row):
        testrow1=test_row
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
        if dum > self.highcutoff1:
            #predicted_price=5
            predicted_price1= self.random_estimator3.predict(test_row)[0] 
            
            x1 = test_row.full_sq
            x1 = x1.reshape(len(x1),1)
            
            predicted_price2= self.Linear_estimator3.predict(x1)[0]
            
        
            predicted_price=0.2*predicted_price1+ 0.8*predicted_price2
            
            
        else:
            if flag==1:
                
                predicted_price1= self.random_estimator2.predict(test_row)[0] 
                x1 = test_row.full_sq
                x1 = x1.reshape(len(x1),1)
                predicted_price2= self.Linear_estimator2.predict(x1)[0]
                predicted_price=0.5*predicted_price1+ 0.5*predicted_price2
            
            else:
                if self.FLAG2==1:
                    testrow1=testrow1.drop(['sub_area', 'full_sq' ,'product_type'],axis=1)
                    testrow1=testrow1.iloc[:,2:20]
                    for fea in testrow1.columns:
                              testrow1[fea].fillna(testrow1[fea].median(),inplace=True)
                    for fea in testrow1.columns:
                              testrow1[fea].fillna(0,inplace=True)
                              
                    pclass=self.TwoClassModel.predict(testrow1)[0]
                
                else:
                       pclass=self.TwoClassModel.predict(test_row)[0]
                       
                if pclass==-1:
                       #predicted_price= 1.5
                       x1 = test_row.full_sq
                       x1 = x1.reshape(len(x1),1)
                       predicted_price= self.Linear_estimator4.predict(x1)[0]
                else:                        
                       predicted_price1= self.random_estimator3.predict(test_row)[0] 
                       x1 = test_row.full_sq
                       x1 = x1.reshape(len(x1),1)
                       predicted_price2= self.Linear_estimator3.predict(x1)[0]
                       predicted_price=.9*predicted_price1+ .1*predicted_price2
                       
        if self.specialFlag==1:
            #if (dum >45) & (dum <90) & (predicted_price > 4):
            #     predicted_price=predicted_price-1.7
            if (dum >100) :
                 predicted_price +=  5    
        return predicted_price

gc.collect()
###############################################################
################################################################################################################
class Mitino():
    #_data = pd.DataFrame
    def __init__(self, train_rows, twoclass):
        
        #### Flags
        self.specialFlag=0 ### this is for special adjustment in predict
        self.FLAG1=0 ### this is 1 for combined s2 s3
        self.FLAG2=0 ### this is 1 for global TwoClass classifier
        
        
        
        self.highcutoff = 150
        self.highcutoff1=150
        self.lowpricecutoff =3.4

        s1=train_rows[train_rows.full_sq > self.highcutoff];
        sa=train_rows[train_rows.full_sq < self.highcutoff];
        s2=sa[(sa.state.isnull()) & (sa.life_sq.isnull())]
        s3a = sa[~((sa.state.isnull()) & (sa.life_sq.isnull()))]
        s3=s3a[s3a.price_doc>self.lowpricecutoff]
        s4=s3a[s3a.price_doc <= self.lowpricecutoff]
        t1=s1.price_doc
        t2=s2.price_doc
        t3=s3.price_doc
        t4=s4.price_doc
        s1=s1.drop(['price_doc','sub_area','product_type'],axis=1)
        s1=s1.iloc[:,2:20]
        for fea in s1.columns:
                s1[fea].fillna(s1[fea].median(), inplace=True)
        for fea in s1.columns:
                s1[fea].fillna(0, inplace=True)
        s2=s2.drop(['price_doc','sub_area','product_type'],axis=1)
        s2=s2.iloc[:,2:20]
        for fea in s2.columns:
                s2[fea].fillna(s2[fea].median(), inplace=True)
        for fea in s2.columns:
                s2[fea].fillna(0, inplace=True)              
        s3=s3.drop(['price_doc','sub_area','product_type'],axis=1)
        s3=s3.iloc[:,2:20]
        for fea in s3.columns:
                s3[fea].fillna(s3[fea].median(), inplace=True)
        for fea in s3.columns:
                s3[fea].fillna(0, inplace=True)
        s4=s4.drop(['price_doc','sub_area','product_type'],axis=1)
        s4=s4.iloc[:,2:20]
        for fea in s4.columns:
                s4[fea].fillna(s4[fea].median(), inplace=True)
        for fea in s4.columns:
                s4[fea].fillna(0, inplace=True)
        
        sa=sa.drop(['price_doc','sub_area','product_type'],axis=1)
        sa=sa.iloc[:,2:20]
        for fea in sa.columns:
                sa[fea].fillna(sa[fea].median(), inplace=True)
        for fea in sa.columns:
                sa[fea].fillna(0, inplace=True)
                
                
        if self.FLAG1==1:
           X23 = pd.concat([s2,s3]) 
           y23 = np.concatenate([t2,t3],axis=0)
           random_estimator23=RandomForestRegressor(random_state=0, n_estimators=500)
           random_estimator23.fit(X23,y23)
           self.random_estimator2=random_estimator23
           self.random_estimator3=random_estimator23
           
           X23r=X23.full_sq
           X23r=X23r.reshape(len(X23r),1)
           y23r=y23.reshape(len(y23), 1)
           Linear_estimator23=LinearRegression()
           Linear_estimator23.fit(X23r, y23r)
           
           self.Linear_estimator2=Linear_estimator23
           self.Linear_estimator3=Linear_estimator23
           
        else:
            
            random_estimator2 = RandomForestRegressor(random_state=0, n_estimators=500)
            random_estimator2.fit(s2,t2)
            self.random_estimator2=random_estimator2
            
            Linear_estimator2 = LinearRegression()
            
            s2r=s2.full_sq
            s2r=s2r.reshape(len(s2r),1)
            t2r=t2
            t2r=t2r.reshape(len(t2r),1)
            Linear_estimator2.fit(s2r, t2r)
            self.Linear_estimator2=Linear_estimator2
            
            
            
            random_estimator3 = RandomForestRegressor(random_state=0, n_estimators=500)
            random_estimator3.fit(s3,t3)
            self.random_estimator3=random_estimator3
            
            Linear_estimator3 = LinearRegression()
            
            s3r=s3.full_sq
            s3r=s3r.reshape(len(s3r),1)
            t3r=t3
            t3r=t3r.reshape(len(t3r),1)
            Linear_estimator3.fit(s3r, t3r)
            self.Linear_estimator3=Linear_estimator3
            
            
            
           
        if self.FLAG2==1:
               self.TwoClassModel=twoclass
        else :
               
                    
                d3=np.ones(len(t3))
                d4=np.ones(len(t4))*-1
                          
                X = pd.concat([s3,s4])
                y = np.concatenate([d3,d4],axis=0)
                self.y=y
                self.X=X
                #print (X)
                #self.decision_clf = svm.SVC()
                TwoClassModel = RandomForestClassifier()
                TwoClassModel.fit(X, y)
                self.TwoClassModel=TwoClassModel
                
            
        Linear_estimator4 = LinearRegression()        
        s4r=s4.full_sq
        s4r=s4r.reshape(len(s4r),1)
        t4r=t4
        t4r=t4r.reshape(len(t4r),1)
        Linear_estimator4.fit(s4r, t4r)
        self.Linear_estimator4=Linear_estimator4             

        
    def predict_price(self,test_row):
        testrow1=test_row
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
        if dum > self.highcutoff1:
            #predicted_price=5
            predicted_price1= self.random_estimator3.predict(test_row)[0] 
            
            x1 = test_row.full_sq
            x1 = x1.reshape(len(x1),1)
            
            predicted_price2= self.Linear_estimator3.predict(x1)[0]
            
        
            predicted_price=0.2*predicted_price1+ 0.8*predicted_price2
            
            
        else:
            if flag==1:
                
                predicted_price1= self.random_estimator2.predict(test_row)[0] 
                x1 = test_row.full_sq
                x1 = x1.reshape(len(x1),1)
                predicted_price2= self.Linear_estimator2.predict(x1)[0]
                predicted_price=0.5*predicted_price1+ 0.5*predicted_price2
            
            else:
                if self.FLAG2==1:
                    testrow1=testrow1.drop(['sub_area', 'full_sq' ,'product_type'],axis=1)
                    testrow1=testrow1.iloc[:,2:20]
                    for fea in testrow1.columns:
                              testrow1[fea].fillna(testrow1[fea].median(),inplace=True)
                    for fea in testrow1.columns:
                              testrow1[fea].fillna(0,inplace=True)
                              
                    pclass=self.TwoClassModel.predict(testrow1)[0]
                
                else:
                       pclass=self.TwoClassModel.predict(test_row)[0]
                       
                if pclass==-1:
                       #predicted_price= 1.5
                       x1 = test_row.full_sq
                       x1 = x1.reshape(len(x1),1)
                       predicted_price= self.Linear_estimator4.predict(x1)[0]
                else:                        
                       predicted_price1= self.random_estimator3.predict(test_row)[0] 
                       x1 = test_row.full_sq
                       x1 = x1.reshape(len(x1),1)
                       predicted_price2= self.Linear_estimator3.predict(x1)[0]
                       predicted_price=1*predicted_price1+ .0*predicted_price2
                       
        if self.specialFlag==1:
            #if (dum >45) & (dum <90) & (predicted_price > 4):
            #     predicted_price=predicted_price-1.7
            if (dum >100) :
                 predicted_price +=  5    
        return predicted_price

gc.collect()
###############################################################
################################################################################################################
class Beskudnikovskoe():
    #_data = pd.DataFrame
    def __init__(self, train_rows, twoclass):
        
        #### Flags
        self.specialFlag=1 ### this is for special adjustment in predict
        self.FLAG1=1 ### this is 1 for combined s2 s3
        self.FLAG2=0 ### this is 1 for global TwoClass classifier
        
        
        
        self.highcutoff = 150
        self.highcutoff1=150
        self.lowpricecutoff =4.8

        s1=train_rows[train_rows.full_sq > self.highcutoff];
        sa=train_rows[train_rows.full_sq < self.highcutoff];
        s2=sa[(sa.state.isnull()) & (sa.life_sq.isnull())]
        s3a = sa[~((sa.state.isnull()) & (sa.life_sq.isnull()))]
        s3=s3a[s3a.price_doc>self.lowpricecutoff]
        s4=s3a[s3a.price_doc <= self.lowpricecutoff]
        t1=s1.price_doc
        t2=s2.price_doc
        t3=s3.price_doc
        t4=s4.price_doc
        s1=s1.drop(['price_doc','sub_area','product_type'],axis=1)
        s1=s1.iloc[:,2:20]
        for fea in s1.columns:
                s1[fea].fillna(s1[fea].median(), inplace=True)
        for fea in s1.columns:
                s1[fea].fillna(0, inplace=True)
        s2=s2.drop(['price_doc','sub_area','product_type'],axis=1)
        s2=s2.iloc[:,2:20]
        for fea in s2.columns:
                s2[fea].fillna(s2[fea].median(), inplace=True)
        for fea in s2.columns:
                s2[fea].fillna(0, inplace=True)              
        s3=s3.drop(['price_doc','sub_area','product_type'],axis=1)
        s3=s3.iloc[:,2:20]
        for fea in s3.columns:
                s3[fea].fillna(s3[fea].median(), inplace=True)
        for fea in s3.columns:
                s3[fea].fillna(0, inplace=True)
        s4=s4.drop(['price_doc','sub_area','product_type'],axis=1)
        s4=s4.iloc[:,2:20]
        for fea in s4.columns:
                s4[fea].fillna(s4[fea].median(), inplace=True)
        for fea in s4.columns:
                s4[fea].fillna(0, inplace=True)
        
        sa=sa.drop(['price_doc','sub_area','product_type'],axis=1)
        sa=sa.iloc[:,2:20]
        for fea in sa.columns:
                sa[fea].fillna(sa[fea].median(), inplace=True)
        for fea in sa.columns:
                sa[fea].fillna(0, inplace=True)
                
                
        if self.FLAG1==1:
           X23 = pd.concat([s2,s3]) 
           y23 = np.concatenate([t2,t3],axis=0)
           random_estimator23=RandomForestRegressor(random_state=0, n_estimators=500)
           random_estimator23.fit(X23,y23)
           self.random_estimator2=random_estimator23
           self.random_estimator3=random_estimator23
           
           X23r=X23.full_sq
           X23r=X23r.reshape(len(X23r),1)
           y23r=y23.reshape(len(y23), 1)
           Linear_estimator23=LinearRegression()
           Linear_estimator23.fit(X23r, y23r)
           
           self.Linear_estimator2=Linear_estimator23
           self.Linear_estimator3=Linear_estimator23
           
        else:
            
            random_estimator2 = RandomForestRegressor(random_state=0, n_estimators=500)
            random_estimator2.fit(s2,t2)
            self.random_estimator2=random_estimator2
            
            Linear_estimator2 = LinearRegression()
            
            s2r=s2.full_sq
            s2r=s2r.reshape(len(s2r),1)
            t2r=t2
            t2r=t2r.reshape(len(t2r),1)
            Linear_estimator2.fit(s2r, t2r)
            self.Linear_estimator2=Linear_estimator2
            
            
            
            random_estimator3 = RandomForestRegressor(random_state=0, n_estimators=500)
            random_estimator3.fit(s3,t3)
            self.random_estimator3=random_estimator3
            
            Linear_estimator3 = LinearRegression()
            
            s3r=s3.full_sq
            s3r=s3r.reshape(len(s3r),1)
            t3r=t3
            t3r=t3r.reshape(len(t3r),1)
            Linear_estimator3.fit(s3r, t3r)
            self.Linear_estimator3=Linear_estimator3
            
            
            
           
        if self.FLAG2==1:
               self.TwoClassModel=twoclass
        else :
               
                    
                d3=np.ones(len(t3))
                d4=np.ones(len(t4))*-1
                          
                X = pd.concat([s3,s4])
                y = np.concatenate([d3,d4],axis=0)
                self.y=y
                self.X=X
                #print (X)
                #self.decision_clf = svm.SVC()
                TwoClassModel = RandomForestClassifier()
                TwoClassModel.fit(X, y)
                self.TwoClassModel=TwoClassModel
                
            
        Linear_estimator4 = LinearRegression()        
        s4r=s4.full_sq
        s4r=s4r.reshape(len(s4r),1)
        t4r=t4
        t4r=t4r.reshape(len(t4r),1)
        Linear_estimator4.fit(s4r, t4r)
        self.Linear_estimator4=Linear_estimator4             

        
    def predict_price(self,test_row):
        testrow1=test_row
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
        if dum > self.highcutoff1:
            #predicted_price=5
            predicted_price1= self.random_estimator3.predict(test_row)[0] 
            
            x1 = test_row.full_sq
            x1 = x1.reshape(len(x1),1)
            
            predicted_price2= self.Linear_estimator3.predict(x1)[0]
            
        
            predicted_price=0.2*predicted_price1+ 0.8*predicted_price2
            
            
        else:
            if flag==1:
                
                predicted_price1= self.random_estimator2.predict(test_row)[0] 
                x1 = test_row.full_sq
                x1 = x1.reshape(len(x1),1)
                predicted_price2= self.Linear_estimator2.predict(x1)[0]
                predicted_price=0.5*predicted_price1+ 0.5*predicted_price2
            
            else:
                if self.FLAG2==1:
                    testrow1=testrow1.drop(['sub_area', 'full_sq' ,'product_type'],axis=1)
                    testrow1=testrow1.iloc[:,2:20]
                    for fea in testrow1.columns:
                              testrow1[fea].fillna(testrow1[fea].median(),inplace=True)
                    for fea in testrow1.columns:
                              testrow1[fea].fillna(0,inplace=True)
                              
                    pclass=self.TwoClassModel.predict(testrow1)[0]
                
                else:
                       pclass=self.TwoClassModel.predict(test_row)[0]
                       
                if pclass==-1:
                       #predicted_price= 1.5
                       x1 = test_row.full_sq
                       x1 = x1.reshape(len(x1),1)
                       predicted_price= self.Linear_estimator4.predict(x1)[0]
                else:                        
                       predicted_price1= self.random_estimator3.predict(test_row)[0] 
                       x1 = test_row.full_sq
                       x1 = x1.reshape(len(x1),1)
                       predicted_price2= self.Linear_estimator3.predict(x1)[0]
                       predicted_price=1*predicted_price1+ .0*predicted_price2
                       
        if self.specialFlag==1:
            #if (dum >45) & (dum <90) & (predicted_price > 4):
            #     predicted_price=predicted_price-1.7
            if (dum >70) :
                 predicted_price -=  1.7    
        return predicted_price

gc.collect()
###############################################################
################################################################################################################
class Vojkovskoe():
    #_data = pd.DataFrame
    def __init__(self, train_rows, twoclass):
        
        #### Flags
        self.specialFlag=0 ### this is for special adjustment in predict
        self.FLAG1=1 ### this is 1 for combined s2 s3
        self.FLAG2=0 ### this is 1 for global TwoClass classifier
        
        
        
        self.highcutoff = 150
        self.highcutoff1=150
        self.lowpricecutoff =4.1

        s1=train_rows[train_rows.full_sq > self.highcutoff];
        sa=train_rows[train_rows.full_sq < self.highcutoff];
        s2=sa[(sa.state.isnull()) & (sa.life_sq.isnull())]
        s3a = sa[~((sa.state.isnull()) & (sa.life_sq.isnull()))]
        s3=s3a[s3a.price_doc>self.lowpricecutoff]
        s4=s3a[s3a.price_doc <= self.lowpricecutoff]
        t1=s1.price_doc
        t2=s2.price_doc
        t3=s3.price_doc
        t4=s4.price_doc
        s1=s1.drop(['price_doc','sub_area','product_type'],axis=1)
        s1=s1.iloc[:,2:20]
        for fea in s1.columns:
                s1[fea].fillna(s1[fea].median(), inplace=True)
        for fea in s1.columns:
                s1[fea].fillna(0, inplace=True)
        s2=s2.drop(['price_doc','sub_area','product_type'],axis=1)
        s2=s2.iloc[:,2:20]
        for fea in s2.columns:
                s2[fea].fillna(s2[fea].median(), inplace=True)
        for fea in s2.columns:
                s2[fea].fillna(0, inplace=True)              
        s3=s3.drop(['price_doc','sub_area','product_type'],axis=1)
        s3=s3.iloc[:,2:20]
        for fea in s3.columns:
                s3[fea].fillna(s3[fea].median(), inplace=True)
        for fea in s3.columns:
                s3[fea].fillna(0, inplace=True)
        s4=s4.drop(['price_doc','sub_area','product_type'],axis=1)
        s4=s4.iloc[:,2:20]
        for fea in s4.columns:
                s4[fea].fillna(s4[fea].median(), inplace=True)
        for fea in s4.columns:
                s4[fea].fillna(0, inplace=True)
        
        sa=sa.drop(['price_doc','sub_area','product_type'],axis=1)
        sa=sa.iloc[:,2:20]
        for fea in sa.columns:
                sa[fea].fillna(sa[fea].median(), inplace=True)
        for fea in sa.columns:
                sa[fea].fillna(0, inplace=True)
                
                
        if self.FLAG1==1:
           X23 = pd.concat([s2,s3]) 
           y23 = np.concatenate([t2,t3],axis=0)
           random_estimator23=RandomForestRegressor(random_state=0, n_estimators=500)
           random_estimator23.fit(X23,y23)
           self.random_estimator2=random_estimator23
           self.random_estimator3=random_estimator23
           
           X23r=X23.full_sq
           X23r=X23r.reshape(len(X23r),1)
           y23r=y23.reshape(len(y23), 1)
           Linear_estimator23=LinearRegression()
           Linear_estimator23.fit(X23r, y23r)
           
           self.Linear_estimator2=Linear_estimator23
           self.Linear_estimator3=Linear_estimator23
           
        else:
            
            random_estimator2 = RandomForestRegressor(random_state=0, n_estimators=500)
            random_estimator2.fit(s2,t2)
            self.random_estimator2=random_estimator2
            
            Linear_estimator2 = LinearRegression()
            
            s2r=s2.full_sq
            s2r=s2r.reshape(len(s2r),1)
            t2r=t2
            t2r=t2r.reshape(len(t2r),1)
            Linear_estimator2.fit(s2r, t2r)
            self.Linear_estimator2=Linear_estimator2
            
            
            
            random_estimator3 = RandomForestRegressor(random_state=0, n_estimators=500)
            random_estimator3.fit(s3,t3)
            self.random_estimator3=random_estimator3
            
            Linear_estimator3 = LinearRegression()
            
            s3r=s3.full_sq
            s3r=s3r.reshape(len(s3r),1)
            t3r=t3
            t3r=t3r.reshape(len(t3r),1)
            Linear_estimator3.fit(s3r, t3r)
            self.Linear_estimator3=Linear_estimator3
            
            
            
           
        if self.FLAG2==1:
               self.TwoClassModel=twoclass
        else :
               
                    
                d3=np.ones(len(t3))
                d4=np.ones(len(t4))*-1
                          
                X = pd.concat([s3,s4])
                y = np.concatenate([d3,d4],axis=0)
                self.y=y
                self.X=X
                #print (X)
                #self.decision_clf = svm.SVC()
                TwoClassModel = RandomForestClassifier()
                TwoClassModel.fit(X, y)
                self.TwoClassModel=TwoClassModel
                
            
        Linear_estimator4 = LinearRegression()        
        s4r=s4.full_sq
        s4r=s4r.reshape(len(s4r),1)
        t4r=t4
        t4r=t4r.reshape(len(t4r),1)
        Linear_estimator4.fit(s4r, t4r)
        self.Linear_estimator4=Linear_estimator4             

        
    def predict_price(self,test_row):
        testrow1=test_row
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
        if dum > self.highcutoff1:
            #predicted_price=5
            predicted_price1= self.random_estimator3.predict(test_row)[0] 
            
            x1 = test_row.full_sq
            x1 = x1.reshape(len(x1),1)
            
            predicted_price2= self.Linear_estimator3.predict(x1)[0]
            
        
            predicted_price=0.2*predicted_price1+ 0.8*predicted_price2
            
            
        else:
            if flag==1:
                
                predicted_price1= self.random_estimator2.predict(test_row)[0] 
                x1 = test_row.full_sq
                x1 = x1.reshape(len(x1),1)
                predicted_price2= self.Linear_estimator2.predict(x1)[0]
                predicted_price=0.5*predicted_price1+ 0.5*predicted_price2
            
            else:
                if self.FLAG2==1:
                    testrow1=testrow1.drop(['sub_area', 'full_sq' ,'product_type'],axis=1)
                    testrow1=testrow1.iloc[:,2:20]
                    for fea in testrow1.columns:
                              testrow1[fea].fillna(testrow1[fea].median(),inplace=True)
                    for fea in testrow1.columns:
                              testrow1[fea].fillna(0,inplace=True)
                              
                    pclass=self.TwoClassModel.predict(testrow1)[0]
                
                else:
                       pclass=self.TwoClassModel.predict(test_row)[0]
                       
                if pclass==-1:
                       #predicted_price= 1.5
                       x1 = test_row.full_sq
                       x1 = x1.reshape(len(x1),1)
                       predicted_price= self.Linear_estimator4.predict(x1)[0]
                else:                        
                       predicted_price1= self.random_estimator3.predict(test_row)[0] 
                       x1 = test_row.full_sq
                       x1 = x1.reshape(len(x1),1)
                       predicted_price2= self.Linear_estimator3.predict(x1)[0]
                       predicted_price=1*predicted_price1+ .0*predicted_price2
                       
        if self.specialFlag==1:
            #if (dum >45) & (dum <90) & (predicted_price > 4):
            #     predicted_price=predicted_price-1.7
            if (dum >70) :
                 predicted_price -=  1.7    
        return predicted_price

gc.collect()
###############################################################
################################################################################################################
class Troickij_okrug():
    #_data = pd.DataFrame
    def __init__(self, train_rows, twoclass):
        
        #### Flags
        self.specialFlag=1 ### this is for special adjustment in predict
        self.FLAG1=0 ### this is 1 for combined s2 s3
        self.FLAG2=0 ### this is 1 for global TwoClass classifier
        
        
        
        self.highcutoff = 150
        self.highcutoff1=150
        self.lowpricecutoff =1.5

        s1=train_rows[train_rows.full_sq > self.highcutoff];
        sa=train_rows[train_rows.full_sq < self.highcutoff];
        s2=sa[(sa.state.isnull()) & (sa.life_sq.isnull())]
        s3a = sa[~((sa.state.isnull()) & (sa.life_sq.isnull()))]
        s3=s3a[s3a.price_doc>self.lowpricecutoff]
        s4=s3a[s3a.price_doc <= self.lowpricecutoff]
        t1=s1.price_doc
        t2=s2.price_doc
        t3=s3.price_doc
        t4=s4.price_doc
        s1=s1.drop(['price_doc','sub_area','product_type'],axis=1)
        s1=s1.iloc[:,2:20]
        for fea in s1.columns:
                s1[fea].fillna(s1[fea].median(), inplace=True)
        for fea in s1.columns:
                s1[fea].fillna(0, inplace=True)
        s2=s2.drop(['price_doc','sub_area','product_type'],axis=1)
        s2=s2.iloc[:,2:20]
        for fea in s2.columns:
                s2[fea].fillna(s2[fea].median(), inplace=True)
        for fea in s2.columns:
                s2[fea].fillna(0, inplace=True)              
        s3=s3.drop(['price_doc','sub_area','product_type'],axis=1)
        s3=s3.iloc[:,2:20]
        for fea in s3.columns:
                s3[fea].fillna(s3[fea].median(), inplace=True)
        for fea in s3.columns:
                s3[fea].fillna(0, inplace=True)
        s4=s4.drop(['price_doc','sub_area','product_type'],axis=1)
        s4=s4.iloc[:,2:20]
        for fea in s4.columns:
                s4[fea].fillna(s4[fea].median(), inplace=True)
        for fea in s4.columns:
                s4[fea].fillna(0, inplace=True)
        
        sa=sa.drop(['price_doc','sub_area','product_type'],axis=1)
        sa=sa.iloc[:,2:20]
        for fea in sa.columns:
                sa[fea].fillna(sa[fea].median(), inplace=True)
        for fea in sa.columns:
                sa[fea].fillna(0, inplace=True)
                
                
        if self.FLAG1==1:
           X23 = pd.concat([s2,s3]) 
           y23 = np.concatenate([t2,t3],axis=0)
           random_estimator23=RandomForestRegressor(random_state=0, n_estimators=500)
           random_estimator23.fit(X23,y23)
           self.random_estimator2=random_estimator23
           self.random_estimator3=random_estimator23
           
           X23r=X23.full_sq
           X23r=X23r.reshape(len(X23r),1)
           y23r=y23.reshape(len(y23), 1)
           Linear_estimator23=LinearRegression()
           Linear_estimator23.fit(X23r, y23r)
           
           self.Linear_estimator2=Linear_estimator23
           self.Linear_estimator3=Linear_estimator23
           
        else:
            
            random_estimator2 = RandomForestRegressor(random_state=0, n_estimators=500)
            random_estimator2.fit(s2,t2)
            self.random_estimator2=random_estimator2
            
            Linear_estimator2 = LinearRegression()
            
            s2r=s2.full_sq
            s2r=s2r.reshape(len(s2r),1)
            t2r=t2
            t2r=t2r.reshape(len(t2r),1)
            Linear_estimator2.fit(s2r, t2r)
            self.Linear_estimator2=Linear_estimator2
            
            
            
            random_estimator3 = RandomForestRegressor(random_state=0, n_estimators=500)
            random_estimator3.fit(s3,t3)
            self.random_estimator3=random_estimator3
            
            Linear_estimator3 = LinearRegression()
            
            s3r=s3.full_sq
            s3r=s3r.reshape(len(s3r),1)
            t3r=t3
            t3r=t3r.reshape(len(t3r),1)
            Linear_estimator3.fit(s3r, t3r)
            self.Linear_estimator3=Linear_estimator3
            
            
            
           
        if self.FLAG2==1:
               self.TwoClassModel=twoclass
        else :
               
                    
                d3=np.ones(len(t3))
                d4=np.ones(len(t4))*-1
                          
                X = pd.concat([s3,s4])
                y = np.concatenate([d3,d4],axis=0)
                self.y=y
                self.X=X
                #print (X)
                #self.decision_clf = svm.SVC()
                TwoClassModel = RandomForestClassifier()
                TwoClassModel.fit(X, y)
                self.TwoClassModel=TwoClassModel
                
            
        Linear_estimator4 = LinearRegression()        
        s4r=s4.full_sq
        s4r=s4r.reshape(len(s4r),1)
        t4r=t4
        t4r=t4r.reshape(len(t4r),1)
        Linear_estimator4.fit(s4r, t4r)
        self.Linear_estimator4=Linear_estimator4             

        
    def predict_price(self,test_row):
        testrow1=test_row
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
        if dum > self.highcutoff1:
            #predicted_price=5
            predicted_price1= self.random_estimator3.predict(test_row)[0] 
            
            x1 = test_row.full_sq
            x1 = x1.reshape(len(x1),1)
            
            predicted_price2= self.Linear_estimator3.predict(x1)[0]
            
        
            predicted_price=0.2*predicted_price1+ 0.8*predicted_price2
            
            
        else:
            if flag==1:
                
                predicted_price1= self.random_estimator2.predict(test_row)[0] 
                x1 = test_row.full_sq
                x1 = x1.reshape(len(x1),1)
                predicted_price2= self.Linear_estimator2.predict(x1)[0]
                predicted_price=0.5*predicted_price1+ 0.5*predicted_price2
            
            else:
                if self.FLAG2==1:
                    testrow1=testrow1.drop(['sub_area', 'full_sq' ,'product_type'],axis=1)
                    testrow1=testrow1.iloc[:,2:20]
                    for fea in testrow1.columns:
                              testrow1[fea].fillna(testrow1[fea].median(),inplace=True)
                    for fea in testrow1.columns:
                              testrow1[fea].fillna(0,inplace=True)
                              
                    pclass=self.TwoClassModel.predict(testrow1)[0]
                
                else:
                       pclass=self.TwoClassModel.predict(test_row)[0]
                       
                if pclass==-1:
                       #predicted_price= 1.5
                       x1 = test_row.full_sq
                       x1 = x1.reshape(len(x1),1)
                       predicted_price= self.Linear_estimator4.predict(x1)[0]
                else:                        
                       predicted_price1= self.random_estimator3.predict(test_row)[0] 
                       x1 = test_row.full_sq
                       x1 = x1.reshape(len(x1),1)
                       predicted_price2= self.Linear_estimator3.predict(x1)[0]
                       predicted_price=.7*predicted_price1+ .3*predicted_price2
                       
        if self.specialFlag==1:
            #if (dum >45) & (dum <90) & (predicted_price > 4):
            #     predicted_price=predicted_price-1.7
            if (dum >200) :
                 predicted_price =10    
        return predicted_price

gc.collect()
###############################################################
################################################################################################################
class Rjazanskij():
    #_data = pd.DataFrame
    def __init__(self, train_rows, twoclass):
        
        #### Flags
        self.specialFlag=1 ### this is for special adjustment in predict
        self.FLAG1=1 ### this is 1 for combined s2 s3
        self.FLAG2=0 ### this is 1 for global TwoClass classifier
        
        
        
        self.highcutoff = 150
        self.highcutoff1=150
        self.lowpricecutoff =3.5

        s1=train_rows[train_rows.full_sq > self.highcutoff];
        sa=train_rows[train_rows.full_sq < self.highcutoff];
        s2=sa[(sa.state.isnull()) & (sa.life_sq.isnull())]
        s3a = sa[~((sa.state.isnull()) & (sa.life_sq.isnull()))]
        s3=s3a[s3a.price_doc>self.lowpricecutoff]
        s4=s3a[s3a.price_doc <= self.lowpricecutoff]
        t1=s1.price_doc
        t2=s2.price_doc
        t3=s3.price_doc
        t4=s4.price_doc
        s1=s1.drop(['price_doc','sub_area','product_type'],axis=1)
        s1=s1.iloc[:,2:20]
        for fea in s1.columns:
                s1[fea].fillna(s1[fea].median(), inplace=True)
        for fea in s1.columns:
                s1[fea].fillna(0, inplace=True)
        s2=s2.drop(['price_doc','sub_area','product_type'],axis=1)
        s2=s2.iloc[:,2:20]
        for fea in s2.columns:
                s2[fea].fillna(s2[fea].median(), inplace=True)
        for fea in s2.columns:
                s2[fea].fillna(0, inplace=True)              
        s3=s3.drop(['price_doc','sub_area','product_type'],axis=1)
        s3=s3.iloc[:,2:20]
        for fea in s3.columns:
                s3[fea].fillna(s3[fea].median(), inplace=True)
        for fea in s3.columns:
                s3[fea].fillna(0, inplace=True)
        s4=s4.drop(['price_doc','sub_area','product_type'],axis=1)
        s4=s4.iloc[:,2:20]
        for fea in s4.columns:
                s4[fea].fillna(s4[fea].median(), inplace=True)
        for fea in s4.columns:
                s4[fea].fillna(0, inplace=True)
        
        sa=sa.drop(['price_doc','sub_area','product_type'],axis=1)
        sa=sa.iloc[:,2:20]
        for fea in sa.columns:
                sa[fea].fillna(sa[fea].median(), inplace=True)
        for fea in sa.columns:
                sa[fea].fillna(0, inplace=True)
                
                
        if self.FLAG1==1:
           X23 = pd.concat([s2,s3]) 
           y23 = np.concatenate([t2,t3],axis=0)
           random_estimator23=RandomForestRegressor(random_state=0, n_estimators=500)
           random_estimator23.fit(X23,y23)
           self.random_estimator2=random_estimator23
           self.random_estimator3=random_estimator23
           
           X23r=X23.full_sq
           X23r=X23r.reshape(len(X23r),1)
           y23r=y23.reshape(len(y23), 1)
           Linear_estimator23=LinearRegression()
           Linear_estimator23.fit(X23r, y23r)
           
           self.Linear_estimator2=Linear_estimator23
           self.Linear_estimator3=Linear_estimator23
           
        else:
            
            random_estimator2 = RandomForestRegressor(random_state=0, n_estimators=500)
            random_estimator2.fit(s2,t2)
            self.random_estimator2=random_estimator2
            
            Linear_estimator2 = LinearRegression()
            
            s2r=s2.full_sq
            s2r=s2r.reshape(len(s2r),1)
            t2r=t2
            t2r=t2r.reshape(len(t2r),1)
            Linear_estimator2.fit(s2r, t2r)
            self.Linear_estimator2=Linear_estimator2
            
            
            
            random_estimator3 = RandomForestRegressor(random_state=0, n_estimators=500)
            random_estimator3.fit(s3,t3)
            self.random_estimator3=random_estimator3
            
            Linear_estimator3 = LinearRegression()
            
            s3r=s3.full_sq
            s3r=s3r.reshape(len(s3r),1)
            t3r=t3
            t3r=t3r.reshape(len(t3r),1)
            Linear_estimator3.fit(s3r, t3r)
            self.Linear_estimator3=Linear_estimator3
            
            
            
           
        if self.FLAG2==1:
               self.TwoClassModel=twoclass
        else :
               
                    
                d3=np.ones(len(t3))
                d4=np.ones(len(t4))*-1
                          
                X = pd.concat([s3,s4])
                y = np.concatenate([d3,d4],axis=0)
                self.y=y
                self.X=X
                #print (X)
                #self.decision_clf = svm.SVC()
                TwoClassModel = RandomForestClassifier()
                TwoClassModel.fit(X, y)
                self.TwoClassModel=TwoClassModel
                
            
        Linear_estimator4 = LinearRegression()        
        s4r=s4.full_sq
        s4r=s4r.reshape(len(s4r),1)
        t4r=t4
        t4r=t4r.reshape(len(t4r),1)
        Linear_estimator4.fit(s4r, t4r)
        self.Linear_estimator4=Linear_estimator4             

        
    def predict_price(self,test_row):
        testrow1=test_row
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
        if dum > self.highcutoff1:
            #predicted_price=5
            predicted_price1= self.random_estimator3.predict(test_row)[0] 
            
            x1 = test_row.full_sq
            x1 = x1.reshape(len(x1),1)
            
            predicted_price2= self.Linear_estimator3.predict(x1)[0]
            
        
            predicted_price=0.2*predicted_price1+ 0.8*predicted_price2
            
            
        else:
            if flag==1:
                
                predicted_price1= self.random_estimator2.predict(test_row)[0] 
                x1 = test_row.full_sq
                x1 = x1.reshape(len(x1),1)
                predicted_price2= self.Linear_estimator2.predict(x1)[0]
                predicted_price=0.5*predicted_price1+ 0.5*predicted_price2
            
            else:
                if self.FLAG2==1:
                    testrow1=testrow1.drop(['sub_area', 'full_sq' ,'product_type'],axis=1)
                    testrow1=testrow1.iloc[:,2:20]
                    for fea in testrow1.columns:
                              testrow1[fea].fillna(testrow1[fea].median(),inplace=True)
                    for fea in testrow1.columns:
                              testrow1[fea].fillna(0,inplace=True)
                              
                    pclass=self.TwoClassModel.predict(testrow1)[0]
                
                else:
                       pclass=self.TwoClassModel.predict(test_row)[0]
                       
                if pclass==-1:
                       #predicted_price= 1.5
                       x1 = test_row.full_sq
                       x1 = x1.reshape(len(x1),1)
                       predicted_price= self.Linear_estimator4.predict(x1)[0]
                else:                        
                       predicted_price1= self.random_estimator3.predict(test_row)[0] 
                       x1 = test_row.full_sq
                       x1 = x1.reshape(len(x1),1)
                       predicted_price2= self.Linear_estimator3.predict(x1)[0]
                       predicted_price=.8*predicted_price1+ .2*predicted_price2
                       
        if self.specialFlag==1:
            #if (dum >45) & (dum <90) & (predicted_price > 4):
            #     predicted_price=predicted_price-1.7
            if (dum >70) :
                 predicted_price -=2.3    
        return predicted_price

gc.collect()
###############################################################
###############################################################################################################
class Severnoe_Butovo():
    #_data = pd.DataFrame
    def __init__(self, train_rows, twoclass):
        
        #### Flags
        self.specialFlag=0 ### this is for special adjustment in predict
        self.FLAG1=1 ### this is 1 for combined s2 s3
        self.FLAG2=0 ### this is 1 for global TwoClass classifier
        
        
        
        self.highcutoff = 150
        self.highcutoff1=150
        self.lowpricecutoff =4.5

        s1=train_rows[train_rows.full_sq > self.highcutoff];
        sa=train_rows[train_rows.full_sq < self.highcutoff];
        s2=sa[(sa.state.isnull()) & (sa.life_sq.isnull())]
        s3a = sa[~((sa.state.isnull()) & (sa.life_sq.isnull()))]
        s3=s3a[s3a.price_doc>self.lowpricecutoff]
        s4=s3a[s3a.price_doc <= self.lowpricecutoff]
        t1=s1.price_doc
        t2=s2.price_doc
        t3=s3.price_doc
        t4=s4.price_doc
        s1=s1.drop(['price_doc','sub_area','product_type'],axis=1)
        s1=s1.iloc[:,2:20]
        for fea in s1.columns:
                s1[fea].fillna(s1[fea].median(), inplace=True)
        for fea in s1.columns:
                s1[fea].fillna(0, inplace=True)
        s2=s2.drop(['price_doc','sub_area','product_type'],axis=1)
        s2=s2.iloc[:,2:20]
        for fea in s2.columns:
                s2[fea].fillna(s2[fea].median(), inplace=True)
        for fea in s2.columns:
                s2[fea].fillna(0, inplace=True)              
        s3=s3.drop(['price_doc','sub_area','product_type'],axis=1)
        s3=s3.iloc[:,2:20]
        for fea in s3.columns:
                s3[fea].fillna(s3[fea].median(), inplace=True)
        for fea in s3.columns:
                s3[fea].fillna(0, inplace=True)
        s4=s4.drop(['price_doc','sub_area','product_type'],axis=1)
        s4=s4.iloc[:,2:20]
        for fea in s4.columns:
                s4[fea].fillna(s4[fea].median(), inplace=True)
        for fea in s4.columns:
                s4[fea].fillna(0, inplace=True)
        
        sa=sa.drop(['price_doc','sub_area','product_type'],axis=1)
        sa=sa.iloc[:,2:20]
        for fea in sa.columns:
                sa[fea].fillna(sa[fea].median(), inplace=True)
        for fea in sa.columns:
                sa[fea].fillna(0, inplace=True)
                
                
        if self.FLAG1==1:
           X23 = pd.concat([s2,s3]) 
           y23 = np.concatenate([t2,t3],axis=0)
           random_estimator23=RandomForestRegressor(random_state=0, n_estimators=500)
           random_estimator23.fit(X23,y23)
           self.random_estimator2=random_estimator23
           self.random_estimator3=random_estimator23
           
           X23r=X23.full_sq
           X23r=X23r.reshape(len(X23r),1)
           y23r=y23.reshape(len(y23), 1)
           Linear_estimator23=LinearRegression()
           Linear_estimator23.fit(X23r, y23r)
           
           self.Linear_estimator2=Linear_estimator23
           self.Linear_estimator3=Linear_estimator23
           
        else:
            
            random_estimator2 = RandomForestRegressor(random_state=0, n_estimators=500)
            random_estimator2.fit(s2,t2)
            self.random_estimator2=random_estimator2
            
            Linear_estimator2 = LinearRegression()
            
            s2r=s2.full_sq
            s2r=s2r.reshape(len(s2r),1)
            t2r=t2
            t2r=t2r.reshape(len(t2r),1)
            Linear_estimator2.fit(s2r, t2r)
            self.Linear_estimator2=Linear_estimator2
            
            
            
            random_estimator3 = RandomForestRegressor(random_state=0, n_estimators=500)
            random_estimator3.fit(s3,t3)
            self.random_estimator3=random_estimator3
            
            Linear_estimator3 = LinearRegression()
            
            s3r=s3.full_sq
            s3r=s3r.reshape(len(s3r),1)
            t3r=t3
            t3r=t3r.reshape(len(t3r),1)
            Linear_estimator3.fit(s3r, t3r)
            self.Linear_estimator3=Linear_estimator3
            
            
            
           
        if self.FLAG2==1:
               self.TwoClassModel=twoclass
        else :
               
                    
                d3=np.ones(len(t3))
                d4=np.ones(len(t4))*-1
                          
                X = pd.concat([s3,s4])
                y = np.concatenate([d3,d4],axis=0)
                self.y=y
                self.X=X
                #print (X)
                #self.decision_clf = svm.SVC()
                TwoClassModel = RandomForestClassifier()
                TwoClassModel.fit(X, y)
                self.TwoClassModel=TwoClassModel
                
            
        Linear_estimator4 = LinearRegression()        
        s4r=s4.full_sq
        s4r=s4r.reshape(len(s4r),1)
        t4r=t4
        t4r=t4r.reshape(len(t4r),1)
        Linear_estimator4.fit(s4r, t4r)
        self.Linear_estimator4=Linear_estimator4             

        
    def predict_price(self,test_row):
        testrow1=test_row
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
        if dum > self.highcutoff1:
            #predicted_price=5
            predicted_price1= self.random_estimator3.predict(test_row)[0] 
            
            x1 = test_row.full_sq
            x1 = x1.reshape(len(x1),1)
            
            predicted_price2= self.Linear_estimator3.predict(x1)[0]
            
        
            predicted_price=0.2*predicted_price1+ 0.8*predicted_price2
            
            
        else:
            if flag==1:
                
                predicted_price1= self.random_estimator2.predict(test_row)[0] 
                x1 = test_row.full_sq
                x1 = x1.reshape(len(x1),1)
                predicted_price2= self.Linear_estimator2.predict(x1)[0]
                predicted_price=0.5*predicted_price1+ 0.5*predicted_price2
            
            else:
                if self.FLAG2==1:
                    testrow1=testrow1.drop(['sub_area', 'full_sq' ,'product_type'],axis=1)
                    testrow1=testrow1.iloc[:,2:20]
                    for fea in testrow1.columns:
                              testrow1[fea].fillna(testrow1[fea].median(),inplace=True)
                    for fea in testrow1.columns:
                              testrow1[fea].fillna(0,inplace=True)
                              
                    pclass=self.TwoClassModel.predict(testrow1)[0]
                
                else:
                       pclass=self.TwoClassModel.predict(test_row)[0]
                       
                if pclass==-1:
                       #predicted_price= 1.5
                       x1 = test_row.full_sq
                       x1 = x1.reshape(len(x1),1)
                       predicted_price= self.Linear_estimator4.predict(x1)[0]
                else:                        
                       predicted_price1= self.random_estimator3.predict(test_row)[0] 
                       x1 = test_row.full_sq
                       x1 = x1.reshape(len(x1),1)
                       predicted_price2= self.Linear_estimator3.predict(x1)[0]
                       predicted_price=.8*predicted_price1+ .2*predicted_price2
                       
        if self.specialFlag==1:
            #if (dum >45) & (dum <90) & (predicted_price > 4):
            #     predicted_price=predicted_price-1.7
            if (dum >70) :
                 predicted_price -=2.3    
        return predicted_price

gc.collect()
###############################################################
################################################################################################################
class Kuncevo():
    #_data = pd.DataFrame
    def __init__(self, train_rows, twoclass):
        
        #### Flags
        self.specialFlag=0 ### this is for special adjustment in predict
        self.FLAG1=1 ### this is 1 for combined s2 s3
        self.FLAG2=0 ### this is 1 for global TwoClass classifier
        
        
        
        self.highcutoff = 150
        self.highcutoff1=150
        self.lowpricecutoff =.99

        s1=train_rows[train_rows.full_sq > self.highcutoff];
        sa=train_rows[train_rows.full_sq < self.highcutoff];
        s2=sa[(sa.state.isnull()) & (sa.life_sq.isnull())]
        s3a = sa[~((sa.state.isnull()) & (sa.life_sq.isnull()))]
        s3=s3a[s3a.price_doc>self.lowpricecutoff]
        s4=s3a[s3a.price_doc <= self.lowpricecutoff]
        t1=s1.price_doc
        t2=s2.price_doc
        t3=s3.price_doc
        t4=s4.price_doc
        s1=s1.drop(['price_doc','sub_area','product_type'],axis=1)
        s1=s1.iloc[:,2:20]
        for fea in s1.columns:
                s1[fea].fillna(s1[fea].median(), inplace=True)
        for fea in s1.columns:
                s1[fea].fillna(0, inplace=True)
        s2=s2.drop(['price_doc','sub_area','product_type'],axis=1)
        s2=s2.iloc[:,2:20]
        for fea in s2.columns:
                s2[fea].fillna(s2[fea].median(), inplace=True)
        for fea in s2.columns:
                s2[fea].fillna(0, inplace=True)              
        s3=s3.drop(['price_doc','sub_area','product_type'],axis=1)
        s3=s3.iloc[:,2:20]
        for fea in s3.columns:
                s3[fea].fillna(s3[fea].median(), inplace=True)
        for fea in s3.columns:
                s3[fea].fillna(0, inplace=True)
        s4=s4.drop(['price_doc','sub_area','product_type'],axis=1)
        s4=s4.iloc[:,2:20]
        for fea in s4.columns:
                s4[fea].fillna(s4[fea].median(), inplace=True)
        for fea in s4.columns:
                s4[fea].fillna(0, inplace=True)
        
        sa=sa.drop(['price_doc','sub_area','product_type'],axis=1)
        sa=sa.iloc[:,2:20]
        for fea in sa.columns:
                sa[fea].fillna(sa[fea].median(), inplace=True)
        for fea in sa.columns:
                sa[fea].fillna(0, inplace=True)
                
                
        if self.FLAG1==1:
           X23 = pd.concat([s2,s3]) 
           y23 = np.concatenate([t2,t3],axis=0)
           random_estimator23=RandomForestRegressor(random_state=0, n_estimators=500)
           random_estimator23.fit(X23,y23)
           self.random_estimator2=random_estimator23
           self.random_estimator3=random_estimator23
           
           X23r=X23.full_sq
           X23r=X23r.reshape(len(X23r),1)
           y23r=y23.reshape(len(y23), 1)
           Linear_estimator23=LinearRegression()
           Linear_estimator23.fit(X23r, y23r)
           
           self.Linear_estimator2=Linear_estimator23
           self.Linear_estimator3=Linear_estimator23
           
        else:
            
            random_estimator2 = RandomForestRegressor(random_state=0, n_estimators=500)
            random_estimator2.fit(s2,t2)
            self.random_estimator2=random_estimator2
            
            Linear_estimator2 = LinearRegression()
            
            s2r=s2.full_sq
            s2r=s2r.reshape(len(s2r),1)
            t2r=t2
            t2r=t2r.reshape(len(t2r),1)
            Linear_estimator2.fit(s2r, t2r)
            self.Linear_estimator2=Linear_estimator2
            
            
            
            random_estimator3 = RandomForestRegressor(random_state=0, n_estimators=500)
            random_estimator3.fit(s3,t3)
            self.random_estimator3=random_estimator3
            
            Linear_estimator3 = LinearRegression()
            
            s3r=s3.full_sq
            s3r=s3r.reshape(len(s3r),1)
            t3r=t3
            t3r=t3r.reshape(len(t3r),1)
            Linear_estimator3.fit(s3r, t3r)
            self.Linear_estimator3=Linear_estimator3
            
            
            
           
        if self.FLAG2==1:
               self.TwoClassModel=twoclass
        else :
               
                    
                d3=np.ones(len(t3))
                d4=np.ones(len(t4))*-1
                          
                X = pd.concat([s3,s4])
                y = np.concatenate([d3,d4],axis=0)
                self.y=y
                self.X=X
                #print (X)
                #self.decision_clf = svm.SVC()
                TwoClassModel = RandomForestClassifier()
                TwoClassModel.fit(X, y)
                self.TwoClassModel=TwoClassModel
                
            
        Linear_estimator4 = LinearRegression()        
        s4r=s4.full_sq
        s4r=s4r.reshape(len(s4r),1)
        t4r=t4
        t4r=t4r.reshape(len(t4r),1)
        Linear_estimator4.fit(s4r, t4r)
        self.Linear_estimator4=Linear_estimator4             

        
    def predict_price(self,test_row):
        testrow1=test_row
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
        if dum > self.highcutoff1:
            #predicted_price=5
            predicted_price1= self.random_estimator3.predict(test_row)[0] 
            
            x1 = test_row.full_sq
            x1 = x1.reshape(len(x1),1)
            
            predicted_price2= self.Linear_estimator3.predict(x1)[0]
            
        
            predicted_price=0.2*predicted_price1+ 0.8*predicted_price2
            
            
        else:
            if flag==1:
                
                predicted_price1= self.random_estimator2.predict(test_row)[0] 
                x1 = test_row.full_sq
                x1 = x1.reshape(len(x1),1)
                predicted_price2= self.Linear_estimator2.predict(x1)[0]
                predicted_price=0.5*predicted_price1+ 0.5*predicted_price2
            
            else:
                if self.FLAG2==1:
                    testrow1=testrow1.drop(['sub_area', 'full_sq' ,'product_type'],axis=1)
                    testrow1=testrow1.iloc[:,2:20]
                    for fea in testrow1.columns:
                              testrow1[fea].fillna(testrow1[fea].median(),inplace=True)
                    for fea in testrow1.columns:
                              testrow1[fea].fillna(0,inplace=True)
                              
                    pclass=self.TwoClassModel.predict(testrow1)[0]
                
                else:
                       pclass=self.TwoClassModel.predict(test_row)[0]
                       
                if pclass==-1:
                       #predicted_price= 1.5
                       x1 = test_row.full_sq
                       x1 = x1.reshape(len(x1),1)
                       predicted_price= self.Linear_estimator4.predict(x1)[0]
                else:                        
                       predicted_price1= self.random_estimator3.predict(test_row)[0] 
                       x1 = test_row.full_sq
                       x1 = x1.reshape(len(x1),1)
                       predicted_price2= self.Linear_estimator3.predict(x1)[0]
                       predicted_price=1*predicted_price1+ .0*predicted_price2
                       
        if self.specialFlag==1:
            #if (dum >45) & (dum <90) & (predicted_price > 4):
            #     predicted_price=predicted_price-1.7
            if (dum >70) :
                 predicted_price -=2.3    
        return predicted_price

gc.collect()
###############################################################
################################################################################################################
class Severnoe():
    #_data = pd.DataFrame
    def __init__(self, train_rows, twoclass):
        
        #### Flags
        self.specialFlag=1 ### this is for special adjustment in predict
        self.FLAG1=1 ### this is 1 for combined s2 s3
        self.FLAG2=0 ### this is 1 for global TwoClass classifier
        
        
        
        self.highcutoff = 150
        self.highcutoff1=150
        self.lowpricecutoff =.99

        s1=train_rows[train_rows.full_sq > self.highcutoff];
        sa=train_rows[train_rows.full_sq < self.highcutoff];
        s2=sa[(sa.state.isnull()) & (sa.life_sq.isnull())]
        s3a = sa[~((sa.state.isnull()) & (sa.life_sq.isnull()))]
        s3=s3a[s3a.price_doc>self.lowpricecutoff]
        s4=s3a[s3a.price_doc <= self.lowpricecutoff]
        t1=s1.price_doc
        t2=s2.price_doc
        t3=s3.price_doc
        t4=s4.price_doc
        s1=s1.drop(['price_doc','sub_area','product_type'],axis=1)
        s1=s1.iloc[:,2:20]
        for fea in s1.columns:
                s1[fea].fillna(s1[fea].median(), inplace=True)
        for fea in s1.columns:
                s1[fea].fillna(0, inplace=True)
        s2=s2.drop(['price_doc','sub_area','product_type'],axis=1)
        s2=s2.iloc[:,2:20]
        for fea in s2.columns:
                s2[fea].fillna(s2[fea].median(), inplace=True)
        for fea in s2.columns:
                s2[fea].fillna(0, inplace=True)              
        s3=s3.drop(['price_doc','sub_area','product_type'],axis=1)
        s3=s3.iloc[:,2:20]
        for fea in s3.columns:
                s3[fea].fillna(s3[fea].median(), inplace=True)
        for fea in s3.columns:
                s3[fea].fillna(0, inplace=True)
        s4=s4.drop(['price_doc','sub_area','product_type'],axis=1)
        s4=s4.iloc[:,2:20]
        for fea in s4.columns:
                s4[fea].fillna(s4[fea].median(), inplace=True)
        for fea in s4.columns:
                s4[fea].fillna(0, inplace=True)
        
        sa=sa.drop(['price_doc','sub_area','product_type'],axis=1)
        sa=sa.iloc[:,2:20]
        for fea in sa.columns:
                sa[fea].fillna(sa[fea].median(), inplace=True)
        for fea in sa.columns:
                sa[fea].fillna(0, inplace=True)
                
                
        if self.FLAG1==1:
           X23 = pd.concat([s2,s3]) 
           y23 = np.concatenate([t2,t3],axis=0)
           random_estimator23=RandomForestRegressor(random_state=0, n_estimators=500)
           random_estimator23.fit(X23,y23)
           self.random_estimator2=random_estimator23
           self.random_estimator3=random_estimator23
           
           X23r=X23.full_sq
           X23r=X23r.reshape(len(X23r),1)
           y23r=y23.reshape(len(y23), 1)
           Linear_estimator23=LinearRegression()
           Linear_estimator23.fit(X23r, y23r)
           
           self.Linear_estimator2=Linear_estimator23
           self.Linear_estimator3=Linear_estimator23
           
        else:
            
            random_estimator2 = RandomForestRegressor(random_state=0, n_estimators=500)
            random_estimator2.fit(s2,t2)
            self.random_estimator2=random_estimator2
            
            Linear_estimator2 = LinearRegression()
            
            s2r=s2.full_sq
            s2r=s2r.reshape(len(s2r),1)
            t2r=t2
            t2r=t2r.reshape(len(t2r),1)
            Linear_estimator2.fit(s2r, t2r)
            self.Linear_estimator2=Linear_estimator2
            
            
            
            random_estimator3 = RandomForestRegressor(random_state=0, n_estimators=500)
            random_estimator3.fit(s3,t3)
            self.random_estimator3=random_estimator3
            
            Linear_estimator3 = LinearRegression()
            
            s3r=s3.full_sq
            s3r=s3r.reshape(len(s3r),1)
            t3r=t3
            t3r=t3r.reshape(len(t3r),1)
            Linear_estimator3.fit(s3r, t3r)
            self.Linear_estimator3=Linear_estimator3
            
            
            
           
        if self.FLAG2==1:
               self.TwoClassModel=twoclass
        else :
               
                    
                d3=np.ones(len(t3))
                d4=np.ones(len(t4))*-1
                          
                X = pd.concat([s3,s4])
                y = np.concatenate([d3,d4],axis=0)
                self.y=y
                self.X=X
                #print (X)
                #self.decision_clf = svm.SVC()
                TwoClassModel = RandomForestClassifier()
                TwoClassModel.fit(X, y)
                self.TwoClassModel=TwoClassModel
                
            
        Linear_estimator4 = LinearRegression()        
        s4r=s4.full_sq
        s4r=s4r.reshape(len(s4r),1)
        t4r=t4
        t4r=t4r.reshape(len(t4r),1)
        Linear_estimator4.fit(s4r, t4r)
        self.Linear_estimator4=Linear_estimator4             

        
    def predict_price(self,test_row):
        testrow1=test_row
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
        if dum > self.highcutoff1:
            #predicted_price=5
            predicted_price1= self.random_estimator3.predict(test_row)[0] 
            
            x1 = test_row.full_sq
            x1 = x1.reshape(len(x1),1)
            
            predicted_price2= self.Linear_estimator3.predict(x1)[0]
            
        
            predicted_price=0.2*predicted_price1+ 0.8*predicted_price2
            
            
        else:
            if flag==1:
                
                predicted_price1= self.random_estimator2.predict(test_row)[0] 
                x1 = test_row.full_sq
                x1 = x1.reshape(len(x1),1)
                predicted_price2= self.Linear_estimator2.predict(x1)[0]
                predicted_price=0.5*predicted_price1+ 0.5*predicted_price2
            
            else:
                if self.FLAG2==1:
                    testrow1=testrow1.drop(['sub_area', 'full_sq' ,'product_type'],axis=1)
                    testrow1=testrow1.iloc[:,2:20]
                    for fea in testrow1.columns:
                              testrow1[fea].fillna(testrow1[fea].median(),inplace=True)
                    for fea in testrow1.columns:
                              testrow1[fea].fillna(0,inplace=True)
                              
                    pclass=self.TwoClassModel.predict(testrow1)[0]
                
                else:
                       pclass=self.TwoClassModel.predict(test_row)[0]
                       
                if pclass==-1:
                       #predicted_price= 1.5
                       x1 = test_row.full_sq
                       x1 = x1.reshape(len(x1),1)
                       predicted_price= self.Linear_estimator4.predict(x1)[0]
                else:                        
                       predicted_price1= self.random_estimator3.predict(test_row)[0] 
                       x1 = test_row.full_sq
                       x1 = x1.reshape(len(x1),1)
                       predicted_price2= self.Linear_estimator3.predict(x1)[0]
                       predicted_price=1*predicted_price1+ .0*predicted_price2
                       
        if self.specialFlag==1:
            #if (dum >45) & (dum <90) & (predicted_price > 4):
            #     predicted_price=predicted_price-1.7
            if (dum >150) :
                 predicted_price =15    
        return predicted_price

gc.collect()
###############################################################
################################################################################################################
class Staroe_Krjukovo():
    #_data = pd.DataFrame
    def __init__(self, train_rows, twoclass):
        
        #### Flags
        self.specialFlag=0 ### this is for special adjustment in predict
        self.FLAG1=1 ### this is 1 for combined s2 s3
        self.FLAG2=0 ### this is 1 for global TwoClass classifier
        
        
        
        self.highcutoff = 150
        self.highcutoff1=150
        self.lowpricecutoff =3.5

        s1=train_rows[train_rows.full_sq > self.highcutoff];
        sa=train_rows[train_rows.full_sq < self.highcutoff];
        s2=sa[(sa.state.isnull()) & (sa.life_sq.isnull())]
        s3a = sa[~((sa.state.isnull()) & (sa.life_sq.isnull()))]
        s3=s3a[s3a.price_doc>self.lowpricecutoff]
        s4=s3a[s3a.price_doc <= self.lowpricecutoff]
        t1=s1.price_doc
        t2=s2.price_doc
        t3=s3.price_doc
        t4=s4.price_doc
        s1=s1.drop(['price_doc','sub_area','product_type'],axis=1)
        s1=s1.iloc[:,2:20]
        for fea in s1.columns:
                s1[fea].fillna(s1[fea].median(), inplace=True)
        for fea in s1.columns:
                s1[fea].fillna(0, inplace=True)
        s2=s2.drop(['price_doc','sub_area','product_type'],axis=1)
        s2=s2.iloc[:,2:20]
        for fea in s2.columns:
                s2[fea].fillna(s2[fea].median(), inplace=True)
        for fea in s2.columns:
                s2[fea].fillna(0, inplace=True)              
        s3=s3.drop(['price_doc','sub_area','product_type'],axis=1)
        s3=s3.iloc[:,2:20]
        for fea in s3.columns:
                s3[fea].fillna(s3[fea].median(), inplace=True)
        for fea in s3.columns:
                s3[fea].fillna(0, inplace=True)
        s4=s4.drop(['price_doc','sub_area','product_type'],axis=1)
        s4=s4.iloc[:,2:20]
        for fea in s4.columns:
                s4[fea].fillna(s4[fea].median(), inplace=True)
        for fea in s4.columns:
                s4[fea].fillna(0, inplace=True)
        
        sa=sa.drop(['price_doc','sub_area','product_type'],axis=1)
        sa=sa.iloc[:,2:20]
        for fea in sa.columns:
                sa[fea].fillna(sa[fea].median(), inplace=True)
        for fea in sa.columns:
                sa[fea].fillna(0, inplace=True)
                
                
        if self.FLAG1==1:
           X23 = pd.concat([s2,s3]) 
           y23 = np.concatenate([t2,t3],axis=0)
           random_estimator23=RandomForestRegressor(random_state=0, n_estimators=500)
           random_estimator23.fit(X23,y23)
           self.random_estimator2=random_estimator23
           self.random_estimator3=random_estimator23
           
           X23r=X23.full_sq
           X23r=X23r.reshape(len(X23r),1)
           y23r=y23.reshape(len(y23), 1)
           Linear_estimator23=LinearRegression()
           Linear_estimator23.fit(X23r, y23r)
           
           self.Linear_estimator2=Linear_estimator23
           self.Linear_estimator3=Linear_estimator23
           
        else:
            
            random_estimator2 = RandomForestRegressor(random_state=0, n_estimators=500)
            random_estimator2.fit(s2,t2)
            self.random_estimator2=random_estimator2
            
            Linear_estimator2 = LinearRegression()
            
            s2r=s2.full_sq
            s2r=s2r.reshape(len(s2r),1)
            t2r=t2
            t2r=t2r.reshape(len(t2r),1)
            Linear_estimator2.fit(s2r, t2r)
            self.Linear_estimator2=Linear_estimator2
            
            
            
            random_estimator3 = RandomForestRegressor(random_state=0, n_estimators=500)
            random_estimator3.fit(s3,t3)
            self.random_estimator3=random_estimator3
            
            Linear_estimator3 = LinearRegression()
            
            s3r=s3.full_sq
            s3r=s3r.reshape(len(s3r),1)
            t3r=t3
            t3r=t3r.reshape(len(t3r),1)
            Linear_estimator3.fit(s3r, t3r)
            self.Linear_estimator3=Linear_estimator3
            
            
            
           
        if self.FLAG2==1:
               self.TwoClassModel=twoclass
        else :
               
                    
                d3=np.ones(len(t3))
                d4=np.ones(len(t4))*-1
                          
                X = pd.concat([s3,s4])
                y = np.concatenate([d3,d4],axis=0)
                self.y=y
                self.X=X
                #print (X)
                #self.decision_clf = svm.SVC()
                TwoClassModel = RandomForestClassifier()
                TwoClassModel.fit(X, y)
                self.TwoClassModel=TwoClassModel
                
            
        Linear_estimator4 = LinearRegression()        
        s4r=s4.full_sq
        s4r=s4r.reshape(len(s4r),1)
        t4r=t4
        t4r=t4r.reshape(len(t4r),1)
        Linear_estimator4.fit(s4r, t4r)
        self.Linear_estimator4=Linear_estimator4             

        
    def predict_price(self,test_row):
        testrow1=test_row
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
        if dum > self.highcutoff1:
            #predicted_price=5
            predicted_price1= self.random_estimator3.predict(test_row)[0] 
            
            x1 = test_row.full_sq
            x1 = x1.reshape(len(x1),1)
            
            predicted_price2= self.Linear_estimator3.predict(x1)[0]
            
        
            predicted_price=0.2*predicted_price1+ 0.8*predicted_price2
            
            
        else:
            if flag==1:
                
                predicted_price1= self.random_estimator2.predict(test_row)[0] 
                x1 = test_row.full_sq
                x1 = x1.reshape(len(x1),1)
                predicted_price2= self.Linear_estimator2.predict(x1)[0]
                predicted_price=0.5*predicted_price1+ 0.5*predicted_price2
            
            else:
                if self.FLAG2==1:
                    testrow1=testrow1.drop(['sub_area', 'full_sq' ,'product_type'],axis=1)
                    testrow1=testrow1.iloc[:,2:20]
                    for fea in testrow1.columns:
                              testrow1[fea].fillna(testrow1[fea].median(),inplace=True)
                    for fea in testrow1.columns:
                              testrow1[fea].fillna(0,inplace=True)
                              
                    pclass=self.TwoClassModel.predict(testrow1)[0]
                
                else:
                       pclass=self.TwoClassModel.predict(test_row)[0]
                       
                if pclass==-1:
                       #predicted_price= 1.5
                       x1 = test_row.full_sq
                       x1 = x1.reshape(len(x1),1)
                       predicted_price= self.Linear_estimator4.predict(x1)[0]
                else:                        
                       predicted_price1= self.random_estimator3.predict(test_row)[0] 
                       x1 = test_row.full_sq
                       x1 = x1.reshape(len(x1),1)
                       predicted_price2= self.Linear_estimator3.predict(x1)[0]
                       predicted_price=1*predicted_price1+ .0*predicted_price2
                       
        if self.specialFlag==1:
            #if (dum >45) & (dum <90) & (predicted_price > 4):
            #     predicted_price=predicted_price-1.7
            if (dum >150) :
                 predicted_price =15    
        return predicted_price

gc.collect()
###############################################################
################################################################################################################
class Golovinskoe():
    #_data = pd.DataFrame
    def __init__(self, train_rows, twoclass):
        
        #### Flags
        self.specialFlag=0 ### this is for special adjustment in predict
        self.FLAG1=1 ### this is 1 for combined s2 s3
        self.FLAG2=0 ### this is 1 for global TwoClass classifier
        
        
        
        self.highcutoff = 150
        self.highcutoff1=150
        self.lowpricecutoff =4.1

        s1=train_rows[train_rows.full_sq > self.highcutoff];
        sa=train_rows[train_rows.full_sq < self.highcutoff];
        s2=sa[(sa.state.isnull()) & (sa.life_sq.isnull())]
        s3a = sa[~((sa.state.isnull()) & (sa.life_sq.isnull()))]
        s3=s3a[s3a.price_doc>self.lowpricecutoff]
        s4=s3a[s3a.price_doc <= self.lowpricecutoff]
        t1=s1.price_doc
        t2=s2.price_doc
        t3=s3.price_doc
        t4=s4.price_doc
        s1=s1.drop(['price_doc','sub_area','product_type'],axis=1)
        s1=s1.iloc[:,2:20]
        for fea in s1.columns:
                s1[fea].fillna(s1[fea].median(), inplace=True)
        for fea in s1.columns:
                s1[fea].fillna(0, inplace=True)
        s2=s2.drop(['price_doc','sub_area','product_type'],axis=1)
        s2=s2.iloc[:,2:20]
        for fea in s2.columns:
                s2[fea].fillna(s2[fea].median(), inplace=True)
        for fea in s2.columns:
                s2[fea].fillna(0, inplace=True)              
        s3=s3.drop(['price_doc','sub_area','product_type'],axis=1)
        s3=s3.iloc[:,2:20]
        for fea in s3.columns:
                s3[fea].fillna(s3[fea].median(), inplace=True)
        for fea in s3.columns:
                s3[fea].fillna(0, inplace=True)
        s4=s4.drop(['price_doc','sub_area','product_type'],axis=1)
        s4=s4.iloc[:,2:20]
        for fea in s4.columns:
                s4[fea].fillna(s4[fea].median(), inplace=True)
        for fea in s4.columns:
                s4[fea].fillna(0, inplace=True)
        
        sa=sa.drop(['price_doc','sub_area','product_type'],axis=1)
        sa=sa.iloc[:,2:20]
        for fea in sa.columns:
                sa[fea].fillna(sa[fea].median(), inplace=True)
        for fea in sa.columns:
                sa[fea].fillna(0, inplace=True)
                
                
        if self.FLAG1==1:
           X23 = pd.concat([s2,s3]) 
           y23 = np.concatenate([t2,t3],axis=0)
           random_estimator23=RandomForestRegressor(random_state=0, n_estimators=500)
           random_estimator23.fit(X23,y23)
           self.random_estimator2=random_estimator23
           self.random_estimator3=random_estimator23
           
           X23r=X23.full_sq
           X23r=X23r.reshape(len(X23r),1)
           y23r=y23.reshape(len(y23), 1)
           Linear_estimator23=LinearRegression()
           Linear_estimator23.fit(X23r, y23r)
           
           self.Linear_estimator2=Linear_estimator23
           self.Linear_estimator3=Linear_estimator23
           
        else:
            
            random_estimator2 = RandomForestRegressor(random_state=0, n_estimators=500)
            random_estimator2.fit(s2,t2)
            self.random_estimator2=random_estimator2
            
            Linear_estimator2 = LinearRegression()
            
            s2r=s2.full_sq
            s2r=s2r.reshape(len(s2r),1)
            t2r=t2
            t2r=t2r.reshape(len(t2r),1)
            Linear_estimator2.fit(s2r, t2r)
            self.Linear_estimator2=Linear_estimator2
            
            
            
            random_estimator3 = RandomForestRegressor(random_state=0, n_estimators=500)
            random_estimator3.fit(s3,t3)
            self.random_estimator3=random_estimator3
            
            Linear_estimator3 = LinearRegression()
            
            s3r=s3.full_sq
            s3r=s3r.reshape(len(s3r),1)
            t3r=t3
            t3r=t3r.reshape(len(t3r),1)
            Linear_estimator3.fit(s3r, t3r)
            self.Linear_estimator3=Linear_estimator3
            
            
            
           
        if self.FLAG2==1:
               self.TwoClassModel=twoclass
        else :
               
                    
                d3=np.ones(len(t3))
                d4=np.ones(len(t4))*-1
                          
                X = pd.concat([s3,s4])
                y = np.concatenate([d3,d4],axis=0)
                self.y=y
                self.X=X
                #print (X)
                #self.decision_clf = svm.SVC()
                TwoClassModel = RandomForestClassifier()
                TwoClassModel.fit(X, y)
                self.TwoClassModel=TwoClassModel
                
            
        Linear_estimator4 = LinearRegression()        
        s4r=s4.full_sq
        s4r=s4r.reshape(len(s4r),1)
        t4r=t4
        t4r=t4r.reshape(len(t4r),1)
        Linear_estimator4.fit(s4r, t4r)
        self.Linear_estimator4=Linear_estimator4             

        
    def predict_price(self,test_row):
        testrow1=test_row
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
        if dum > self.highcutoff1:
            #predicted_price=5
            predicted_price1= self.random_estimator3.predict(test_row)[0] 
            
            x1 = test_row.full_sq
            x1 = x1.reshape(len(x1),1)
            
            predicted_price2= self.Linear_estimator3.predict(x1)[0]
            
        
            predicted_price=0.2*predicted_price1+ 0.8*predicted_price2
            
            
        else:
            if flag==1:
                
                predicted_price1= self.random_estimator2.predict(test_row)[0] 
                x1 = test_row.full_sq
                x1 = x1.reshape(len(x1),1)
                predicted_price2= self.Linear_estimator2.predict(x1)[0]
                predicted_price=0.5*predicted_price1+ 0.5*predicted_price2
            
            else:
                if self.FLAG2==1:
                    testrow1=testrow1.drop(['sub_area', 'full_sq' ,'product_type'],axis=1)
                    testrow1=testrow1.iloc[:,2:20]
                    for fea in testrow1.columns:
                              testrow1[fea].fillna(testrow1[fea].median(),inplace=True)
                    for fea in testrow1.columns:
                              testrow1[fea].fillna(0,inplace=True)
                              
                    pclass=self.TwoClassModel.predict(testrow1)[0]
                
                else:
                       pclass=self.TwoClassModel.predict(test_row)[0]
                       
                if pclass==-1:
                       #predicted_price= 1.5
                       x1 = test_row.full_sq
                       x1 = x1.reshape(len(x1),1)
                       predicted_price= self.Linear_estimator4.predict(x1)[0]
                else:                        
                       predicted_price1= self.random_estimator3.predict(test_row)[0] 
                       x1 = test_row.full_sq
                       x1 = x1.reshape(len(x1),1)
                       predicted_price2= self.Linear_estimator3.predict(x1)[0]
                       predicted_price=.8*predicted_price1+ .2*predicted_price2
                       
        if self.specialFlag==1:
            #if (dum >45) & (dum <90) & (predicted_price > 4):
            #     predicted_price=predicted_price-1.7
            if (dum >150) :
                 predicted_price =15    
        return predicted_price

gc.collect()
###############################################################
################################################################################################################
class Konkovo():
    #_data = pd.DataFrame
    def __init__(self, train_rows, twoclass):
        
        #### Flags
        self.specialFlag=0 ### this is for special adjustment in predict
        self.FLAG1=1 ### this is 1 for combined s2 s3
        self.FLAG2=0 ### this is 1 for global TwoClass classifier
        
        
        
        self.highcutoff = 150
        self.highcutoff1=150
        self.lowpricecutoff =3.5

        s1=train_rows[train_rows.full_sq > self.highcutoff];
        sa=train_rows[train_rows.full_sq < self.highcutoff];
        s2=sa[(sa.state.isnull()) & (sa.life_sq.isnull())]
        s3a = sa[~((sa.state.isnull()) & (sa.life_sq.isnull()))]
        s3=s3a[s3a.price_doc>self.lowpricecutoff]
        s4=s3a[s3a.price_doc <= self.lowpricecutoff]
        t1=s1.price_doc
        t2=s2.price_doc
        t3=s3.price_doc
        t4=s4.price_doc
        s1=s1.drop(['price_doc','sub_area','product_type'],axis=1)
        s1=s1.iloc[:,2:20]
        for fea in s1.columns:
                s1[fea].fillna(s1[fea].median(), inplace=True)
        for fea in s1.columns:
                s1[fea].fillna(0, inplace=True)
        s2=s2.drop(['price_doc','sub_area','product_type'],axis=1)
        s2=s2.iloc[:,2:20]
        for fea in s2.columns:
                s2[fea].fillna(s2[fea].median(), inplace=True)
        for fea in s2.columns:
                s2[fea].fillna(0, inplace=True)              
        s3=s3.drop(['price_doc','sub_area','product_type'],axis=1)
        s3=s3.iloc[:,2:20]
        for fea in s3.columns:
                s3[fea].fillna(s3[fea].median(), inplace=True)
        for fea in s3.columns:
                s3[fea].fillna(0, inplace=True)
        s4=s4.drop(['price_doc','sub_area','product_type'],axis=1)
        s4=s4.iloc[:,2:20]
        for fea in s4.columns:
                s4[fea].fillna(s4[fea].median(), inplace=True)
        for fea in s4.columns:
                s4[fea].fillna(0, inplace=True)
        
        sa=sa.drop(['price_doc','sub_area','product_type'],axis=1)
        sa=sa.iloc[:,2:20]
        for fea in sa.columns:
                sa[fea].fillna(sa[fea].median(), inplace=True)
        for fea in sa.columns:
                sa[fea].fillna(0, inplace=True)
                
                
        if self.FLAG1==1:
           X23 = pd.concat([s2,s3]) 
           y23 = np.concatenate([t2,t3],axis=0)
           random_estimator23=RandomForestRegressor(random_state=0, n_estimators=500)
           random_estimator23.fit(X23,y23)
           self.random_estimator2=random_estimator23
           self.random_estimator3=random_estimator23
           
           X23r=X23.full_sq
           X23r=X23r.reshape(len(X23r),1)
           y23r=y23.reshape(len(y23), 1)
           Linear_estimator23=LinearRegression()
           Linear_estimator23.fit(X23r, y23r)
           
           self.Linear_estimator2=Linear_estimator23
           self.Linear_estimator3=Linear_estimator23
           
        else:
            
            random_estimator2 = RandomForestRegressor(random_state=0, n_estimators=500)
            random_estimator2.fit(s2,t2)
            self.random_estimator2=random_estimator2
            
            Linear_estimator2 = LinearRegression()
            
            s2r=s2.full_sq
            s2r=s2r.reshape(len(s2r),1)
            t2r=t2
            t2r=t2r.reshape(len(t2r),1)
            Linear_estimator2.fit(s2r, t2r)
            self.Linear_estimator2=Linear_estimator2
            
            
            
            random_estimator3 = RandomForestRegressor(random_state=0, n_estimators=500)
            random_estimator3.fit(s3,t3)
            self.random_estimator3=random_estimator3
            
            Linear_estimator3 = LinearRegression()
            
            s3r=s3.full_sq
            s3r=s3r.reshape(len(s3r),1)
            t3r=t3
            t3r=t3r.reshape(len(t3r),1)
            Linear_estimator3.fit(s3r, t3r)
            self.Linear_estimator3=Linear_estimator3
            
            
            
           
        if self.FLAG2==1:
               self.TwoClassModel=twoclass
        else :
               
                    
                d3=np.ones(len(t3))
                d4=np.ones(len(t4))*-1
                          
                X = pd.concat([s3,s4])
                y = np.concatenate([d3,d4],axis=0)
                self.y=y
                self.X=X
                #print (X)
                #self.decision_clf = svm.SVC()
                TwoClassModel = RandomForestClassifier()
                TwoClassModel.fit(X, y)
                self.TwoClassModel=TwoClassModel
                
            
        Linear_estimator4 = LinearRegression()        
        s4r=s4.full_sq
        s4r=s4r.reshape(len(s4r),1)
        t4r=t4
        t4r=t4r.reshape(len(t4r),1)
        Linear_estimator4.fit(s4r, t4r)
        self.Linear_estimator4=Linear_estimator4             

        
    def predict_price(self,test_row):
        testrow1=test_row
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
        if dum > self.highcutoff1:
            #predicted_price=5
            predicted_price1= self.random_estimator3.predict(test_row)[0] 
            
            x1 = test_row.full_sq
            x1 = x1.reshape(len(x1),1)
            
            predicted_price2= self.Linear_estimator3.predict(x1)[0]
            
        
            predicted_price=0.2*predicted_price1+ 0.8*predicted_price2
            
            
        else:
            if flag==1:
                
                predicted_price1= self.random_estimator2.predict(test_row)[0] 
                x1 = test_row.full_sq
                x1 = x1.reshape(len(x1),1)
                predicted_price2= self.Linear_estimator2.predict(x1)[0]
                predicted_price=0.5*predicted_price1+ 0.5*predicted_price2
            
            else:
                if self.FLAG2==1:
                    testrow1=testrow1.drop(['sub_area', 'full_sq' ,'product_type'],axis=1)
                    testrow1=testrow1.iloc[:,2:20]
                    for fea in testrow1.columns:
                              testrow1[fea].fillna(testrow1[fea].median(),inplace=True)
                    for fea in testrow1.columns:
                              testrow1[fea].fillna(0,inplace=True)
                              
                    pclass=self.TwoClassModel.predict(testrow1)[0]
                
                else:
                       pclass=self.TwoClassModel.predict(test_row)[0]
                       
                if pclass==-1:
                       #predicted_price= 1.5
                       x1 = test_row.full_sq
                       x1 = x1.reshape(len(x1),1)
                       predicted_price= self.Linear_estimator4.predict(x1)[0]
                else:                        
                       predicted_price1= self.random_estimator3.predict(test_row)[0] 
                       x1 = test_row.full_sq
                       x1 = x1.reshape(len(x1),1)
                       predicted_price2= self.Linear_estimator3.predict(x1)[0]
                       predicted_price=.8*predicted_price1+ .2*predicted_price2
                       
        if self.specialFlag==1:
            #if (dum >45) & (dum <90) & (predicted_price > 4):
            #     predicted_price=predicted_price-1.7
            if (dum >150) :
                 predicted_price =15    
        return predicted_price

gc.collect()
###############################################################
################################################################################################################
class Kosino_Uhtomskoe():
    #_data = pd.DataFrame
    def __init__(self, train_rows, twoclass):
        
        #### Flags
        self.specialFlag=0 ### this is for special adjustment in predict
        self.FLAG1=1 ### this is 1 for combined s2 s3
        self.FLAG2=0 ### this is 1 for global TwoClass classifier
        
        
        
        self.highcutoff = 150
        self.highcutoff1=150
        self.lowpricecutoff =3.5

        s1=train_rows[train_rows.full_sq > self.highcutoff];
        sa=train_rows[train_rows.full_sq < self.highcutoff];
        s2=sa[(sa.state.isnull()) & (sa.life_sq.isnull())]
        s3a = sa[~((sa.state.isnull()) & (sa.life_sq.isnull()))]
        s3=s3a[s3a.price_doc>self.lowpricecutoff]
        s4=s3a[s3a.price_doc <= self.lowpricecutoff]
        t1=s1.price_doc
        t2=s2.price_doc
        t3=s3.price_doc
        t4=s4.price_doc
        s1=s1.drop(['price_doc','sub_area','product_type'],axis=1)
        s1=s1.iloc[:,2:20]
        for fea in s1.columns:
                s1[fea].fillna(s1[fea].median(), inplace=True)
        for fea in s1.columns:
                s1[fea].fillna(0, inplace=True)
        s2=s2.drop(['price_doc','sub_area','product_type'],axis=1)
        s2=s2.iloc[:,2:20]
        for fea in s2.columns:
                s2[fea].fillna(s2[fea].median(), inplace=True)
        for fea in s2.columns:
                s2[fea].fillna(0, inplace=True)              
        s3=s3.drop(['price_doc','sub_area','product_type'],axis=1)
        s3=s3.iloc[:,2:20]
        for fea in s3.columns:
                s3[fea].fillna(s3[fea].median(), inplace=True)
        for fea in s3.columns:
                s3[fea].fillna(0, inplace=True)
        s4=s4.drop(['price_doc','sub_area','product_type'],axis=1)
        s4=s4.iloc[:,2:20]
        for fea in s4.columns:
                s4[fea].fillna(s4[fea].median(), inplace=True)
        for fea in s4.columns:
                s4[fea].fillna(0, inplace=True)
        
        sa=sa.drop(['price_doc','sub_area','product_type'],axis=1)
        sa=sa.iloc[:,2:20]
        for fea in sa.columns:
                sa[fea].fillna(sa[fea].median(), inplace=True)
        for fea in sa.columns:
                sa[fea].fillna(0, inplace=True)
                
                
        if self.FLAG1==1:
           X23 = pd.concat([s2,s3]) 
           y23 = np.concatenate([t2,t3],axis=0)
           random_estimator23=RandomForestRegressor(random_state=0, n_estimators=500)
           random_estimator23.fit(X23,y23)
           self.random_estimator2=random_estimator23
           self.random_estimator3=random_estimator23
           
           X23r=X23.full_sq
           X23r=X23r.reshape(len(X23r),1)
           y23r=y23.reshape(len(y23), 1)
           Linear_estimator23=LinearRegression()
           Linear_estimator23.fit(X23r, y23r)
           
           self.Linear_estimator2=Linear_estimator23
           self.Linear_estimator3=Linear_estimator23
           
        else:
            
            random_estimator2 = RandomForestRegressor(random_state=0, n_estimators=500)
            random_estimator2.fit(s2,t2)
            self.random_estimator2=random_estimator2
            
            Linear_estimator2 = LinearRegression()
            
            s2r=s2.full_sq
            s2r=s2r.reshape(len(s2r),1)
            t2r=t2
            t2r=t2r.reshape(len(t2r),1)
            Linear_estimator2.fit(s2r, t2r)
            self.Linear_estimator2=Linear_estimator2
            
            
            
            random_estimator3 = RandomForestRegressor(random_state=0, n_estimators=500)
            random_estimator3.fit(s3,t3)
            self.random_estimator3=random_estimator3
            
            Linear_estimator3 = LinearRegression()
            
            s3r=s3.full_sq
            s3r=s3r.reshape(len(s3r),1)
            t3r=t3
            t3r=t3r.reshape(len(t3r),1)
            Linear_estimator3.fit(s3r, t3r)
            self.Linear_estimator3=Linear_estimator3
            
            
            
           
        if self.FLAG2==1:
               self.TwoClassModel=twoclass
        else :
               
                    
                d3=np.ones(len(t3))
                d4=np.ones(len(t4))*-1
                          
                X = pd.concat([s3,s4])
                y = np.concatenate([d3,d4],axis=0)
                self.y=y
                self.X=X
                #print (X)
                #self.decision_clf = svm.SVC()
                TwoClassModel = RandomForestClassifier()
                TwoClassModel.fit(X, y)
                self.TwoClassModel=TwoClassModel
                
            
        Linear_estimator4 = LinearRegression()        
        s4r=s4.full_sq
        s4r=s4r.reshape(len(s4r),1)
        t4r=t4
        t4r=t4r.reshape(len(t4r),1)
        Linear_estimator4.fit(s4r, t4r)
        self.Linear_estimator4=Linear_estimator4             

        
    def predict_price(self,test_row):
        testrow1=test_row
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
        if dum > self.highcutoff1:
            #predicted_price=5
            predicted_price1= self.random_estimator3.predict(test_row)[0] 
            
            x1 = test_row.full_sq
            x1 = x1.reshape(len(x1),1)
            
            predicted_price2= self.Linear_estimator3.predict(x1)[0]
            
        
            predicted_price=0.2*predicted_price1+ 0.8*predicted_price2
            
            
        else:
            if flag==1:
                
                predicted_price1= self.random_estimator2.predict(test_row)[0] 
                x1 = test_row.full_sq
                x1 = x1.reshape(len(x1),1)
                predicted_price2= self.Linear_estimator2.predict(x1)[0]
                predicted_price=0.5*predicted_price1+ 0.5*predicted_price2
            
            else:
                if self.FLAG2==1:
                    testrow1=testrow1.drop(['sub_area', 'full_sq' ,'product_type'],axis=1)
                    testrow1=testrow1.iloc[:,2:20]
                    for fea in testrow1.columns:
                              testrow1[fea].fillna(testrow1[fea].median(),inplace=True)
                    for fea in testrow1.columns:
                              testrow1[fea].fillna(0,inplace=True)
                              
                    pclass=self.TwoClassModel.predict(testrow1)[0]
                
                else:
                       pclass=self.TwoClassModel.predict(test_row)[0]
                       
                if pclass==-1:
                       #predicted_price= 1.5
                       x1 = test_row.full_sq
                       x1 = x1.reshape(len(x1),1)
                       predicted_price= self.Linear_estimator4.predict(x1)[0]
                else:                        
                       predicted_price1= self.random_estimator3.predict(test_row)[0] 
                       x1 = test_row.full_sq
                       x1 = x1.reshape(len(x1),1)
                       predicted_price2= self.Linear_estimator3.predict(x1)[0]
                       predicted_price=.6*predicted_price1+ .4*predicted_price2
                       
        if self.specialFlag==1:
            #if (dum >45) & (dum <90) & (predicted_price > 4):
            #     predicted_price=predicted_price-1.7
            if (dum >150) :
                 predicted_price =15    
        return predicted_price

gc.collect()
###############################################################
################################################################################################################
class Veshnjaki():
    #_data = pd.DataFrame
    def __init__(self, train_rows, twoclass):
        
        #### Flags
        self.specialFlag=1 ### this is for special adjustment in predict
        self.FLAG1=1 ### this is 1 for combined s2 s3
        self.FLAG2=0 ### this is 1 for global TwoClass classifier
        
        
        
        self.highcutoff = 150
        self.highcutoff1=150
        self.lowpricecutoff =4.8

        s1=train_rows[train_rows.full_sq > self.highcutoff];
        sa=train_rows[train_rows.full_sq < self.highcutoff];
        s2=sa[(sa.state.isnull()) & (sa.life_sq.isnull())]
        s3a = sa[~((sa.state.isnull()) & (sa.life_sq.isnull()))]
        s3=s3a[s3a.price_doc>self.lowpricecutoff]
        s4=s3a[s3a.price_doc <= self.lowpricecutoff]
        t1=s1.price_doc
        t2=s2.price_doc
        t3=s3.price_doc
        t4=s4.price_doc
        s1=s1.drop(['price_doc','sub_area','product_type'],axis=1)
        s1=s1.iloc[:,2:20]
        for fea in s1.columns:
                s1[fea].fillna(s1[fea].median(), inplace=True)
        for fea in s1.columns:
                s1[fea].fillna(0, inplace=True)
        s2=s2.drop(['price_doc','sub_area','product_type'],axis=1)
        s2=s2.iloc[:,2:20]
        for fea in s2.columns:
                s2[fea].fillna(s2[fea].median(), inplace=True)
        for fea in s2.columns:
                s2[fea].fillna(0, inplace=True)              
        s3=s3.drop(['price_doc','sub_area','product_type'],axis=1)
        s3=s3.iloc[:,2:20]
        for fea in s3.columns:
                s3[fea].fillna(s3[fea].median(), inplace=True)
        for fea in s3.columns:
                s3[fea].fillna(0, inplace=True)
        s4=s4.drop(['price_doc','sub_area','product_type'],axis=1)
        s4=s4.iloc[:,2:20]
        for fea in s4.columns:
                s4[fea].fillna(s4[fea].median(), inplace=True)
        for fea in s4.columns:
                s4[fea].fillna(0, inplace=True)
        
        sa=sa.drop(['price_doc','sub_area','product_type'],axis=1)
        sa=sa.iloc[:,2:20]
        for fea in sa.columns:
                sa[fea].fillna(sa[fea].median(), inplace=True)
        for fea in sa.columns:
                sa[fea].fillna(0, inplace=True)
                
                
        if self.FLAG1==1:
           X23 = pd.concat([s2,s3]) 
           y23 = np.concatenate([t2,t3],axis=0)
           random_estimator23=RandomForestRegressor(random_state=0, n_estimators=500)
           random_estimator23.fit(X23,y23)
           self.random_estimator2=random_estimator23
           self.random_estimator3=random_estimator23
           
           X23r=X23.full_sq
           X23r=X23r.reshape(len(X23r),1)
           y23r=y23.reshape(len(y23), 1)
           Linear_estimator23=LinearRegression()
           Linear_estimator23.fit(X23r, y23r)
           
           self.Linear_estimator2=Linear_estimator23
           self.Linear_estimator3=Linear_estimator23
           
        else:
            
            random_estimator2 = RandomForestRegressor(random_state=0, n_estimators=500)
            random_estimator2.fit(s2,t2)
            self.random_estimator2=random_estimator2
            
            Linear_estimator2 = LinearRegression()
            
            s2r=s2.full_sq
            s2r=s2r.reshape(len(s2r),1)
            t2r=t2
            t2r=t2r.reshape(len(t2r),1)
            Linear_estimator2.fit(s2r, t2r)
            self.Linear_estimator2=Linear_estimator2
            
            
            
            random_estimator3 = RandomForestRegressor(random_state=0, n_estimators=500)
            random_estimator3.fit(s3,t3)
            self.random_estimator3=random_estimator3
            
            Linear_estimator3 = LinearRegression()
            
            s3r=s3.full_sq
            s3r=s3r.reshape(len(s3r),1)
            t3r=t3
            t3r=t3r.reshape(len(t3r),1)
            Linear_estimator3.fit(s3r, t3r)
            self.Linear_estimator3=Linear_estimator3
            
            
            
           
        if self.FLAG2==1:
               self.TwoClassModel=twoclass
        else :
               
                    
                d3=np.ones(len(t3))
                d4=np.ones(len(t4))*-1
                          
                X = pd.concat([s3,s4])
                y = np.concatenate([d3,d4],axis=0)
                self.y=y
                self.X=X
                #print (X)
                #self.decision_clf = svm.SVC()
                TwoClassModel = RandomForestClassifier()
                TwoClassModel.fit(X, y)
                self.TwoClassModel=TwoClassModel
                
            
        Linear_estimator4 = LinearRegression()        
        s4r=s4.full_sq
        s4r=s4r.reshape(len(s4r),1)
        t4r=t4
        t4r=t4r.reshape(len(t4r),1)
        Linear_estimator4.fit(s4r, t4r)
        self.Linear_estimator4=Linear_estimator4             

        
    def predict_price(self,test_row):
        testrow1=test_row
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
        if dum > self.highcutoff1:
            #predicted_price=5
            predicted_price1= self.random_estimator3.predict(test_row)[0] 
            
            x1 = test_row.full_sq
            x1 = x1.reshape(len(x1),1)
            
            predicted_price2= self.Linear_estimator3.predict(x1)[0]
            
        
            predicted_price=0.2*predicted_price1+ 0.8*predicted_price2
            
            
        else:
            if flag==1:
                
                predicted_price1= self.random_estimator2.predict(test_row)[0] 
                x1 = test_row.full_sq
                x1 = x1.reshape(len(x1),1)
                predicted_price2= self.Linear_estimator2.predict(x1)[0]
                predicted_price=0.5*predicted_price1+ 0.5*predicted_price2
            
            else:
                if self.FLAG2==1:
                    testrow1=testrow1.drop(['sub_area', 'full_sq' ,'product_type'],axis=1)
                    testrow1=testrow1.iloc[:,2:20]
                    for fea in testrow1.columns:
                              testrow1[fea].fillna(testrow1[fea].median(),inplace=True)
                    for fea in testrow1.columns:
                              testrow1[fea].fillna(0,inplace=True)
                              
                    pclass=self.TwoClassModel.predict(testrow1)[0]
                
                else:
                       pclass=self.TwoClassModel.predict(test_row)[0]
                       
                if pclass==-1:
                       #predicted_price= 1.5
                       x1 = test_row.full_sq
                       x1 = x1.reshape(len(x1),1)
                       predicted_price= self.Linear_estimator4.predict(x1)[0]
                else:                        
                       predicted_price1= self.random_estimator3.predict(test_row)[0] 
                       x1 = test_row.full_sq
                       x1 = x1.reshape(len(x1),1)
                       predicted_price2= self.Linear_estimator3.predict(x1)[0]
                       predicted_price=.8*predicted_price1+ .2*predicted_price2
                       
        if self.specialFlag==1:
            if (dum >35) & (dum <90) & (predicted_price < 4):
                 predicted_price=predicted_price + 3
            #if (dum >150) :
            #     predicted_price =15    
        return predicted_price

gc.collect()
###############################################################
################################################################################################################
class Horoshevo_Mnevniki():
    #_data = pd.DataFrame
    def __init__(self, train_rows, twoclass):
        
        #### Flags
        self.specialFlag=1 ### this is for special adjustment in predict
        self.FLAG1=1 ### this is 1 for combined s2 s3
        self.FLAG2=0 ### this is 1 for global TwoClass classifier
        
        
        
        self.highcutoff = 150
        self.highcutoff1=150
        self.lowpricecutoff =4.8

        s1=train_rows[train_rows.full_sq > self.highcutoff];
        sa=train_rows[train_rows.full_sq < self.highcutoff];
        s2=sa[(sa.state.isnull()) & (sa.life_sq.isnull())]
        s3a = sa[~((sa.state.isnull()) & (sa.life_sq.isnull()))]
        s3=s3a[s3a.price_doc>self.lowpricecutoff]
        s4=s3a[s3a.price_doc <= self.lowpricecutoff]
        t1=s1.price_doc
        t2=s2.price_doc
        t3=s3.price_doc
        t4=s4.price_doc
        s1=s1.drop(['price_doc','sub_area','product_type'],axis=1)
        s1=s1.iloc[:,2:20]
        for fea in s1.columns:
                s1[fea].fillna(s1[fea].median(), inplace=True)
        for fea in s1.columns:
                s1[fea].fillna(0, inplace=True)
        s2=s2.drop(['price_doc','sub_area','product_type'],axis=1)
        s2=s2.iloc[:,2:20]
        for fea in s2.columns:
                s2[fea].fillna(s2[fea].median(), inplace=True)
        for fea in s2.columns:
                s2[fea].fillna(0, inplace=True)              
        s3=s3.drop(['price_doc','sub_area','product_type'],axis=1)
        s3=s3.iloc[:,2:20]
        for fea in s3.columns:
                s3[fea].fillna(s3[fea].median(), inplace=True)
        for fea in s3.columns:
                s3[fea].fillna(0, inplace=True)
        s4=s4.drop(['price_doc','sub_area','product_type'],axis=1)
        s4=s4.iloc[:,2:20]
        for fea in s4.columns:
                s4[fea].fillna(s4[fea].median(), inplace=True)
        for fea in s4.columns:
                s4[fea].fillna(0, inplace=True)
        
        sa=sa.drop(['price_doc','sub_area','product_type'],axis=1)
        sa=sa.iloc[:,2:20]
        for fea in sa.columns:
                sa[fea].fillna(sa[fea].median(), inplace=True)
        for fea in sa.columns:
                sa[fea].fillna(0, inplace=True)
                
                
        if self.FLAG1==1:
           X23 = pd.concat([s2,s3]) 
           y23 = np.concatenate([t2,t3],axis=0)
           random_estimator23=RandomForestRegressor(random_state=0, n_estimators=500)
           random_estimator23.fit(X23,y23)
           self.random_estimator2=random_estimator23
           self.random_estimator3=random_estimator23
           
           X23r=X23.full_sq
           X23r=X23r.reshape(len(X23r),1)
           y23r=y23.reshape(len(y23), 1)
           Linear_estimator23=LinearRegression()
           Linear_estimator23.fit(X23r, y23r)
           
           self.Linear_estimator2=Linear_estimator23
           self.Linear_estimator3=Linear_estimator23
           
        else:
            
            random_estimator2 = RandomForestRegressor(random_state=0, n_estimators=500)
            random_estimator2.fit(s2,t2)
            self.random_estimator2=random_estimator2
            
            Linear_estimator2 = LinearRegression()
            
            s2r=s2.full_sq
            s2r=s2r.reshape(len(s2r),1)
            t2r=t2
            t2r=t2r.reshape(len(t2r),1)
            Linear_estimator2.fit(s2r, t2r)
            self.Linear_estimator2=Linear_estimator2
            
            
            
            random_estimator3 = RandomForestRegressor(random_state=0, n_estimators=500)
            random_estimator3.fit(s3,t3)
            self.random_estimator3=random_estimator3
            
            Linear_estimator3 = LinearRegression()
            
            s3r=s3.full_sq
            s3r=s3r.reshape(len(s3r),1)
            t3r=t3
            t3r=t3r.reshape(len(t3r),1)
            Linear_estimator3.fit(s3r, t3r)
            self.Linear_estimator3=Linear_estimator3
            
            
            
           
        if self.FLAG2==1:
               self.TwoClassModel=twoclass
        else :
               
                    
                d3=np.ones(len(t3))
                d4=np.ones(len(t4))*-1
                          
                X = pd.concat([s3,s4])
                y = np.concatenate([d3,d4],axis=0)
                self.y=y
                self.X=X
                #print (X)
                #self.decision_clf = svm.SVC()
                TwoClassModel = RandomForestClassifier()
                TwoClassModel.fit(X, y)
                self.TwoClassModel=TwoClassModel
                
            
        Linear_estimator4 = LinearRegression()        
        s4r=s4.full_sq
        s4r=s4r.reshape(len(s4r),1)
        t4r=t4
        t4r=t4r.reshape(len(t4r),1)
        Linear_estimator4.fit(s4r, t4r)
        self.Linear_estimator4=Linear_estimator4             

        
    def predict_price(self,test_row):
        testrow1=test_row
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
        if dum > self.highcutoff1:
            #predicted_price=5
            predicted_price1= self.random_estimator3.predict(test_row)[0] 
            
            x1 = test_row.full_sq
            x1 = x1.reshape(len(x1),1)
            
            predicted_price2= self.Linear_estimator3.predict(x1)[0]
            
        
            predicted_price=0.2*predicted_price1+ 0.8*predicted_price2
            
            
        else:
            if flag==1:
                
                predicted_price1= self.random_estimator2.predict(test_row)[0] 
                x1 = test_row.full_sq
                x1 = x1.reshape(len(x1),1)
                predicted_price2= self.Linear_estimator2.predict(x1)[0]
                predicted_price=0.5*predicted_price1+ 0.5*predicted_price2
            
            else:
                if self.FLAG2==1:
                    testrow1=testrow1.drop(['sub_area', 'full_sq' ,'product_type'],axis=1)
                    testrow1=testrow1.iloc[:,2:20]
                    for fea in testrow1.columns:
                              testrow1[fea].fillna(testrow1[fea].median(),inplace=True)
                    for fea in testrow1.columns:
                              testrow1[fea].fillna(0,inplace=True)
                              
                    pclass=self.TwoClassModel.predict(testrow1)[0]
                
                else:
                       pclass=self.TwoClassModel.predict(test_row)[0]
                       
                if pclass==-1:
                       #predicted_price= 1.5
                       x1 = test_row.full_sq
                       x1 = x1.reshape(len(x1),1)
                       predicted_price= self.Linear_estimator4.predict(x1)[0]
                else:                        
                       predicted_price1= self.random_estimator3.predict(test_row)[0] 
                       x1 = test_row.full_sq
                       x1 = x1.reshape(len(x1),1)
                       predicted_price2= self.Linear_estimator3.predict(x1)[0]
                       predicted_price=.8*predicted_price1+ .2*predicted_price2
                       
        if self.specialFlag==1:
            if (dum >120) & (dum <600) & (predicted_price > 13):
                 predicted_price=predicted_price + 3
            #if (dum >150) :
            #     predicted_price =15    
        return predicted_price


################################################################################################################
class Marino():
    #_data = pd.DataFrame
    def __init__(self, train_rows, twoclass):
        
        #### Flags
        self.specialFlag=0 ### this is for special adjustment in predict
        self.FLAG1=1 ### this is 1 for combined s2 s3
        self.FLAG2=0 ### this is 1 for global TwoClass classifier
        
        
        
        self.highcutoff = 150
        self.highcutoff1=150
        self.lowpricecutoff =3.5

        s1=train_rows[train_rows.full_sq > self.highcutoff];
        sa=train_rows[train_rows.full_sq < self.highcutoff];
        s2=sa[(sa.state.isnull()) & (sa.life_sq.isnull())]
        s3a = sa[~((sa.state.isnull()) & (sa.life_sq.isnull()))]
        s3=s3a[s3a.price_doc>self.lowpricecutoff]
        s4=s3a[s3a.price_doc <= self.lowpricecutoff]
        t1=s1.price_doc
        t2=s2.price_doc
        t3=s3.price_doc
        t4=s4.price_doc
        s1=s1.drop(['price_doc','sub_area','product_type'],axis=1)
        s1=s1.iloc[:,2:20]
        for fea in s1.columns:
                s1[fea].fillna(s1[fea].median(), inplace=True)
        for fea in s1.columns:
                s1[fea].fillna(0, inplace=True)
        s2=s2.drop(['price_doc','sub_area','product_type'],axis=1)
        s2=s2.iloc[:,2:20]
        for fea in s2.columns:
                s2[fea].fillna(s2[fea].median(), inplace=True)
        for fea in s2.columns:
                s2[fea].fillna(0, inplace=True)              
        s3=s3.drop(['price_doc','sub_area','product_type'],axis=1)
        s3=s3.iloc[:,2:20]
        for fea in s3.columns:
                s3[fea].fillna(s3[fea].median(), inplace=True)
        for fea in s3.columns:
                s3[fea].fillna(0, inplace=True)
        s4=s4.drop(['price_doc','sub_area','product_type'],axis=1)
        s4=s4.iloc[:,2:20]
        for fea in s4.columns:
                s4[fea].fillna(s4[fea].median(), inplace=True)
        for fea in s4.columns:
                s4[fea].fillna(0, inplace=True)
        
        sa=sa.drop(['price_doc','sub_area','product_type'],axis=1)
        sa=sa.iloc[:,2:20]
        for fea in sa.columns:
                sa[fea].fillna(sa[fea].median(), inplace=True)
        for fea in sa.columns:
                sa[fea].fillna(0, inplace=True)
                
                
        if self.FLAG1==1:
           X23 = pd.concat([s2,s3]) 
           y23 = np.concatenate([t2,t3],axis=0)
           random_estimator23=RandomForestRegressor(random_state=0, n_estimators=500)
           random_estimator23.fit(X23,y23)
           self.random_estimator2=random_estimator23
           self.random_estimator3=random_estimator23
           
           X23r=X23.full_sq
           X23r=X23r.reshape(len(X23r),1)
           y23r=y23.reshape(len(y23), 1)
           Linear_estimator23=LinearRegression()
           Linear_estimator23.fit(X23r, y23r)
           
           self.Linear_estimator2=Linear_estimator23
           self.Linear_estimator3=Linear_estimator23
           
        else:
            
            random_estimator2 = RandomForestRegressor(random_state=0, n_estimators=500)
            random_estimator2.fit(s2,t2)
            self.random_estimator2=random_estimator2
            
            Linear_estimator2 = LinearRegression()
            
            s2r=s2.full_sq
            s2r=s2r.reshape(len(s2r),1)
            t2r=t2
            t2r=t2r.reshape(len(t2r),1)
            Linear_estimator2.fit(s2r, t2r)
            self.Linear_estimator2=Linear_estimator2
            
            
            
            random_estimator3 = RandomForestRegressor(random_state=0, n_estimators=500)
            random_estimator3.fit(s3,t3)
            self.random_estimator3=random_estimator3
            
            Linear_estimator3 = LinearRegression()
            
            s3r=s3.full_sq
            s3r=s3r.reshape(len(s3r),1)
            t3r=t3
            t3r=t3r.reshape(len(t3r),1)
            Linear_estimator3.fit(s3r, t3r)
            self.Linear_estimator3=Linear_estimator3
            
            
            
           
        if self.FLAG2==1:
               self.TwoClassModel=twoclass
        else :
               
                    
                d3=np.ones(len(t3))
                d4=np.ones(len(t4))*-1
                          
                X = pd.concat([s3,s4])
                y = np.concatenate([d3,d4],axis=0)
                self.y=y
                self.X=X
                #print (X)
                #self.decision_clf = svm.SVC()
                TwoClassModel = RandomForestClassifier()
                TwoClassModel.fit(X, y)
                self.TwoClassModel=TwoClassModel
                
            
        Linear_estimator4 = LinearRegression()        
        s4r=s4.full_sq
        s4r=s4r.reshape(len(s4r),1)
        t4r=t4
        t4r=t4r.reshape(len(t4r),1)
        Linear_estimator4.fit(s4r, t4r)
        self.Linear_estimator4=Linear_estimator4             

        
    def predict_price(self,test_row):
        testrow1=test_row
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
        if dum > self.highcutoff1:
            #predicted_price=5
            predicted_price1= self.random_estimator3.predict(test_row)[0] 
            
            x1 = test_row.full_sq
            x1 = x1.reshape(len(x1),1)
            
            predicted_price2= self.Linear_estimator3.predict(x1)[0]
            
        
            predicted_price=0.2*predicted_price1+ 0.8*predicted_price2
            
            
        else:
            if flag==1:
                
                predicted_price1= self.random_estimator2.predict(test_row)[0] 
                x1 = test_row.full_sq
                x1 = x1.reshape(len(x1),1)
                predicted_price2= self.Linear_estimator2.predict(x1)[0]
                predicted_price=0.5*predicted_price1+ 0.5*predicted_price2
            
            else:
                if self.FLAG2==1:
                    testrow1=testrow1.drop(['sub_area', 'full_sq' ,'product_type'],axis=1)
                    testrow1=testrow1.iloc[:,2:20]
                    for fea in testrow1.columns:
                              testrow1[fea].fillna(testrow1[fea].median(),inplace=True)
                    for fea in testrow1.columns:
                              testrow1[fea].fillna(0,inplace=True)
                              
                    pclass=self.TwoClassModel.predict(testrow1)[0]
                
                else:
                       pclass=self.TwoClassModel.predict(test_row)[0]
                       
                if pclass==-1:
                       #predicted_price= 1.5
                       x1 = test_row.full_sq
                       x1 = x1.reshape(len(x1),1)
                       predicted_price= self.Linear_estimator4.predict(x1)[0]
                else:                        
                       predicted_price1= self.random_estimator3.predict(test_row)[0] 
                       x1 = test_row.full_sq
                       x1 = x1.reshape(len(x1),1)
                       predicted_price2= self.Linear_estimator3.predict(x1)[0]
                       predicted_price=.8*predicted_price1+ .2*predicted_price2
                       
        if self.specialFlag==1:
            if (dum >120) & (dum <600) & (predicted_price > 13):
                 predicted_price=predicted_price + 3
            #if (dum >150) :
            #     predicted_price =15    
        return predicted_price


gc.collect()
###############################################################
################################################################################################################
class Pechatniki():
    #_data = pd.DataFrame
    def __init__(self, train_rows, twoclass):
        
        #### Flags
        self.specialFlag=1 ### this is for special adjustment in predict
        self.FLAG1=1 ### this is 1 for combined s2 s3
        self.FLAG2=0 ### this is 1 for global TwoClass classifier
        
        
        
        self.highcutoff = 150
        self.highcutoff1=150
        self.lowpricecutoff =3.5

        s1=train_rows[train_rows.full_sq > self.highcutoff];
        sa=train_rows[train_rows.full_sq < self.highcutoff];
        s2=sa[(sa.state.isnull()) & (sa.life_sq.isnull())]
        s3a = sa[~((sa.state.isnull()) & (sa.life_sq.isnull()))]
        s3=s3a[s3a.price_doc>self.lowpricecutoff]
        s4=s3a[s3a.price_doc <= self.lowpricecutoff]
        t1=s1.price_doc
        t2=s2.price_doc
        t3=s3.price_doc
        t4=s4.price_doc
        s1=s1.drop(['price_doc','sub_area','product_type'],axis=1)
        s1=s1.iloc[:,2:20]
        for fea in s1.columns:
                s1[fea].fillna(s1[fea].median(), inplace=True)
        for fea in s1.columns:
                s1[fea].fillna(0, inplace=True)
        s2=s2.drop(['price_doc','sub_area','product_type'],axis=1)
        s2=s2.iloc[:,2:20]
        for fea in s2.columns:
                s2[fea].fillna(s2[fea].median(), inplace=True)
        for fea in s2.columns:
                s2[fea].fillna(0, inplace=True)              
        s3=s3.drop(['price_doc','sub_area','product_type'],axis=1)
        s3=s3.iloc[:,2:20]
        for fea in s3.columns:
                s3[fea].fillna(s3[fea].median(), inplace=True)
        for fea in s3.columns:
                s3[fea].fillna(0, inplace=True)
        s4=s4.drop(['price_doc','sub_area','product_type'],axis=1)
        s4=s4.iloc[:,2:20]
        for fea in s4.columns:
                s4[fea].fillna(s4[fea].median(), inplace=True)
        for fea in s4.columns:
                s4[fea].fillna(0, inplace=True)
        
        sa=sa.drop(['price_doc','sub_area','product_type'],axis=1)
        sa=sa.iloc[:,2:20]
        for fea in sa.columns:
                sa[fea].fillna(sa[fea].median(), inplace=True)
        for fea in sa.columns:
                sa[fea].fillna(0, inplace=True)
                
                
        if self.FLAG1==1:
           X23 = pd.concat([s2,s3]) 
           y23 = np.concatenate([t2,t3],axis=0)
           random_estimator23=RandomForestRegressor(random_state=0, n_estimators=500)
           random_estimator23.fit(X23,y23)
           self.random_estimator2=random_estimator23
           self.random_estimator3=random_estimator23
           
           X23r=X23.full_sq
           X23r=X23r.reshape(len(X23r),1)
           y23r=y23.reshape(len(y23), 1)
           Linear_estimator23=LinearRegression()
           Linear_estimator23.fit(X23r, y23r)
           
           self.Linear_estimator2=Linear_estimator23
           self.Linear_estimator3=Linear_estimator23
           
        else:
            
            random_estimator2 = RandomForestRegressor(random_state=0, n_estimators=500)
            random_estimator2.fit(s2,t2)
            self.random_estimator2=random_estimator2
            
            Linear_estimator2 = LinearRegression()
            
            s2r=s2.full_sq
            s2r=s2r.reshape(len(s2r),1)
            t2r=t2
            t2r=t2r.reshape(len(t2r),1)
            Linear_estimator2.fit(s2r, t2r)
            self.Linear_estimator2=Linear_estimator2
            
            
            
            random_estimator3 = RandomForestRegressor(random_state=0, n_estimators=500)
            random_estimator3.fit(s3,t3)
            self.random_estimator3=random_estimator3
            
            Linear_estimator3 = LinearRegression()
            
            s3r=s3.full_sq
            s3r=s3r.reshape(len(s3r),1)
            t3r=t3
            t3r=t3r.reshape(len(t3r),1)
            Linear_estimator3.fit(s3r, t3r)
            self.Linear_estimator3=Linear_estimator3
            
            
            
           
        if self.FLAG2==1:
               self.TwoClassModel=twoclass
        else :
               
                    
                d3=np.ones(len(t3))
                d4=np.ones(len(t4))*-1
                          
                X = pd.concat([s3,s4])
                y = np.concatenate([d3,d4],axis=0)
                self.y=y
                self.X=X
                #print (X)
                #self.decision_clf = svm.SVC()
                TwoClassModel = RandomForestClassifier()
                TwoClassModel.fit(X, y)
                self.TwoClassModel=TwoClassModel
                
            
        Linear_estimator4 = LinearRegression()        
        s4r=s4.full_sq
        s4r=s4r.reshape(len(s4r),1)
        t4r=t4
        t4r=t4r.reshape(len(t4r),1)
        Linear_estimator4.fit(s4r, t4r)
        self.Linear_estimator4=Linear_estimator4             

        
    def predict_price(self,test_row):
        testrow1=test_row
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
        if dum > self.highcutoff1:
            #predicted_price=5
            predicted_price1= self.random_estimator3.predict(test_row)[0] 
            
            x1 = test_row.full_sq
            x1 = x1.reshape(len(x1),1)
            
            predicted_price2= self.Linear_estimator3.predict(x1)[0]
            
        
            predicted_price=0.2*predicted_price1+ 0.8*predicted_price2
            
            
        else:
            if flag==1:
                
                predicted_price1= self.random_estimator2.predict(test_row)[0] 
                x1 = test_row.full_sq
                x1 = x1.reshape(len(x1),1)
                predicted_price2= self.Linear_estimator2.predict(x1)[0]
                predicted_price=0.5*predicted_price1+ 0.5*predicted_price2
            
            else:
                if self.FLAG2==1:
                    testrow1=testrow1.drop(['sub_area', 'full_sq' ,'product_type'],axis=1)
                    testrow1=testrow1.iloc[:,2:20]
                    for fea in testrow1.columns:
                              testrow1[fea].fillna(testrow1[fea].median(),inplace=True)
                    for fea in testrow1.columns:
                              testrow1[fea].fillna(0,inplace=True)
                              
                    pclass=self.TwoClassModel.predict(testrow1)[0]
                
                else:
                       pclass=self.TwoClassModel.predict(test_row)[0]
                       
                if pclass==-1:
                       #predicted_price= 1.5
                       x1 = test_row.full_sq
                       x1 = x1.reshape(len(x1),1)
                       predicted_price= self.Linear_estimator4.predict(x1)[0]
                else:                        
                       predicted_price1= self.random_estimator3.predict(test_row)[0] 
                       x1 = test_row.full_sq
                       x1 = x1.reshape(len(x1),1)
                       predicted_price2= self.Linear_estimator3.predict(x1)[0]
                       predicted_price=.8*predicted_price1+ .2*predicted_price2
                       
        if self.specialFlag==1:
            if (dum >70) & (dum <600) & (predicted_price > 10):
                 predicted_price=predicted_price + 3
            #if (dum >150) :
            #     predicted_price =15    
        return predicted_price


gc.collect()
###############################################################
################################################################################################################
class Tekstilshhiki():
    #_data = pd.DataFrame
    def __init__(self, train_rows, twoclass):
        
        #### Flags
        self.specialFlag=0 ### this is for special adjustment in predict
        self.FLAG1=1 ### this is 1 for combined s2 s3
        self.FLAG2=1 ### this is 1 for global TwoClass classifier
        
        
        
        self.highcutoff = 150
        self.highcutoff1=150
        self.lowpricecutoff =3.5

        s1=train_rows[train_rows.full_sq > self.highcutoff];
        sa=train_rows[train_rows.full_sq < self.highcutoff];
        s2=sa[(sa.state.isnull()) & (sa.life_sq.isnull())]
        s3a = sa[~((sa.state.isnull()) & (sa.life_sq.isnull()))]
        s3=s3a[s3a.price_doc>self.lowpricecutoff]
        s4=s3a[s3a.price_doc <= self.lowpricecutoff]
        t1=s1.price_doc
        t2=s2.price_doc
        t3=s3.price_doc
        t4=s4.price_doc
        s1=s1.drop(['price_doc','sub_area','product_type'],axis=1)
        s1=s1.iloc[:,2:20]
        for fea in s1.columns:
                s1[fea].fillna(s1[fea].median(), inplace=True)
        for fea in s1.columns:
                s1[fea].fillna(0, inplace=True)
        s2=s2.drop(['price_doc','sub_area','product_type'],axis=1)
        s2=s2.iloc[:,2:20]
        for fea in s2.columns:
                s2[fea].fillna(s2[fea].median(), inplace=True)
        for fea in s2.columns:
                s2[fea].fillna(0, inplace=True)              
        s3=s3.drop(['price_doc','sub_area','product_type'],axis=1)
        s3=s3.iloc[:,2:20]
        for fea in s3.columns:
                s3[fea].fillna(s3[fea].median(), inplace=True)
        for fea in s3.columns:
                s3[fea].fillna(0, inplace=True)
        s4=s4.drop(['price_doc','sub_area','product_type'],axis=1)
        s4=s4.iloc[:,2:20]
        for fea in s4.columns:
                s4[fea].fillna(s4[fea].median(), inplace=True)
        for fea in s4.columns:
                s4[fea].fillna(0, inplace=True)
        
        sa=sa.drop(['price_doc','sub_area','product_type'],axis=1)
        sa=sa.iloc[:,2:20]
        for fea in sa.columns:
                sa[fea].fillna(sa[fea].median(), inplace=True)
        for fea in sa.columns:
                sa[fea].fillna(0, inplace=True)
                
                
        if self.FLAG1==1:
           X23 = pd.concat([s2,s3]) 
           y23 = np.concatenate([t2,t3],axis=0)
           random_estimator23=RandomForestRegressor(random_state=0, n_estimators=500)
           random_estimator23.fit(X23,y23)
           self.random_estimator2=random_estimator23
           self.random_estimator3=random_estimator23
           
           X23r=X23.full_sq
           X23r=X23r.reshape(len(X23r),1)
           y23r=y23.reshape(len(y23), 1)
           Linear_estimator23=LinearRegression()
           Linear_estimator23.fit(X23r, y23r)
           
           self.Linear_estimator2=Linear_estimator23
           self.Linear_estimator3=Linear_estimator23
           
        else:
            
            random_estimator2 = RandomForestRegressor(random_state=0, n_estimators=500)
            random_estimator2.fit(s2,t2)
            self.random_estimator2=random_estimator2
            
            Linear_estimator2 = LinearRegression()
            
            s2r=s2.full_sq
            s2r=s2r.reshape(len(s2r),1)
            t2r=t2
            t2r=t2r.reshape(len(t2r),1)
            Linear_estimator2.fit(s2r, t2r)
            self.Linear_estimator2=Linear_estimator2
            
            
            
            random_estimator3 = RandomForestRegressor(random_state=0, n_estimators=500)
            random_estimator3.fit(s3,t3)
            self.random_estimator3=random_estimator3
            
            Linear_estimator3 = LinearRegression()
            
            s3r=s3.full_sq
            s3r=s3r.reshape(len(s3r),1)
            t3r=t3
            t3r=t3r.reshape(len(t3r),1)
            Linear_estimator3.fit(s3r, t3r)
            self.Linear_estimator3=Linear_estimator3
            
            
            
           
        if self.FLAG2==1:
               self.TwoClassModel=twoclass
        else :
               
                    
                d3=np.ones(len(t3))
                d4=np.ones(len(t4))*-1
                          
                X = pd.concat([s3,s4])
                y = np.concatenate([d3,d4],axis=0)
                self.y=y
                self.X=X
                #print (X)
                #self.decision_clf = svm.SVC()
                TwoClassModel = RandomForestClassifier()
                TwoClassModel.fit(X, y)
                self.TwoClassModel=TwoClassModel
                
            
        Linear_estimator4 = LinearRegression()        
        s4r=s4.full_sq
        s4r=s4r.reshape(len(s4r),1)
        t4r=t4
        t4r=t4r.reshape(len(t4r),1)
        Linear_estimator4.fit(s4r, t4r)
        self.Linear_estimator4=Linear_estimator4             

        
    def predict_price(self,test_row):
        testrow1=test_row
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
        if dum > self.highcutoff1:
            #predicted_price=5
            predicted_price1= self.random_estimator3.predict(test_row)[0] 
            
            x1 = test_row.full_sq
            x1 = x1.reshape(len(x1),1)
            
            predicted_price2= self.Linear_estimator3.predict(x1)[0]
            
        
            predicted_price=0.2*predicted_price1+ 0.8*predicted_price2
            
            
        else:
            if flag==1:
                
                predicted_price1= self.random_estimator2.predict(test_row)[0] 
                x1 = test_row.full_sq
                x1 = x1.reshape(len(x1),1)
                predicted_price2= self.Linear_estimator2.predict(x1)[0]
                predicted_price=0.5*predicted_price1+ 0.5*predicted_price2
            
            else:
                if self.FLAG2==1:
                    testrow1=testrow1.drop(['sub_area', 'full_sq' ,'product_type'],axis=1)
                    testrow1=testrow1.iloc[:,2:20]
                    for fea in testrow1.columns:
                              testrow1[fea].fillna(testrow1[fea].median(),inplace=True)
                    for fea in testrow1.columns:
                              testrow1[fea].fillna(0,inplace=True)
                              
                    pclass=self.TwoClassModel.predict(testrow1)[0]
                
                else:
                       pclass=self.TwoClassModel.predict(test_row)[0]
                       
                if pclass==-1:
                       #predicted_price= 1.5
                       x1 = test_row.full_sq
                       x1 = x1.reshape(len(x1),1)
                       predicted_price= self.Linear_estimator4.predict(x1)[0]
                else:                        
                       predicted_price1= self.random_estimator3.predict(test_row)[0] 
                       x1 = test_row.full_sq
                       x1 = x1.reshape(len(x1),1)
                       predicted_price2= self.Linear_estimator3.predict(x1)[0]
                       predicted_price=.9*predicted_price1+ .1*predicted_price2
                       
        if self.specialFlag==1:
            if (dum >70) & (dum <600) & (predicted_price > 10):
                 predicted_price=predicted_price + 3
            #if (dum >150) :
            #     predicted_price =15    
        return predicted_price


gc.collect()
###############################################################
################################################################################################################
class Solncevo():
    #_data = pd.DataFrame
    def __init__(self, train_rows, twoclass):
        
        #### Flags
        self.specialFlag=1 ### this is for special adjustment in predict
        self.FLAG1=0 ### this is 1 for combined s2 s3
        self.FLAG2=0 ### this is 1 for global TwoClass classifier
        
        
        
        self.highcutoff = 150
        self.highcutoff1=150
        self.lowpricecutoff =3.2

        s1=train_rows[train_rows.full_sq > self.highcutoff];
        sa=train_rows[train_rows.full_sq < self.highcutoff];
        s2=sa[(sa.state.isnull()) & (sa.life_sq.isnull())]
        s3a = sa[~((sa.state.isnull()) & (sa.life_sq.isnull()))]
        s3=s3a[s3a.price_doc>self.lowpricecutoff]
        s4=s3a[s3a.price_doc <= self.lowpricecutoff]
        t1=s1.price_doc
        t2=s2.price_doc
        t3=s3.price_doc
        t4=s4.price_doc
        s1=s1.drop(['price_doc','sub_area','product_type'],axis=1)
        s1=s1.iloc[:,2:20]
        for fea in s1.columns:
                s1[fea].fillna(s1[fea].median(), inplace=True)
        for fea in s1.columns:
                s1[fea].fillna(0, inplace=True)
        s2=s2.drop(['price_doc','sub_area','product_type'],axis=1)
        s2=s2.iloc[:,2:20]
        for fea in s2.columns:
                s2[fea].fillna(s2[fea].median(), inplace=True)
        for fea in s2.columns:
                s2[fea].fillna(0, inplace=True)              
        s3=s3.drop(['price_doc','sub_area','product_type'],axis=1)
        s3=s3.iloc[:,2:20]
        for fea in s3.columns:
                s3[fea].fillna(s3[fea].median(), inplace=True)
        for fea in s3.columns:
                s3[fea].fillna(0, inplace=True)
        s4=s4.drop(['price_doc','sub_area','product_type'],axis=1)
        s4=s4.iloc[:,2:20]
        for fea in s4.columns:
                s4[fea].fillna(s4[fea].median(), inplace=True)
        for fea in s4.columns:
                s4[fea].fillna(0, inplace=True)
        
        sa=sa.drop(['price_doc','sub_area','product_type'],axis=1)
        sa=sa.iloc[:,2:20]
        for fea in sa.columns:
                sa[fea].fillna(sa[fea].median(), inplace=True)
        for fea in sa.columns:
                sa[fea].fillna(0, inplace=True)
                
                
        if self.FLAG1==1:
           X23 = pd.concat([s2,s3]) 
           y23 = np.concatenate([t2,t3],axis=0)
           random_estimator23=RandomForestRegressor(random_state=0, n_estimators=500)
           random_estimator23.fit(X23,y23)
           self.random_estimator2=random_estimator23
           self.random_estimator3=random_estimator23
           
           X23r=X23.full_sq
           X23r=X23r.reshape(len(X23r),1)
           y23r=y23.reshape(len(y23), 1)
           Linear_estimator23=LinearRegression()
           Linear_estimator23.fit(X23r, y23r)
           
           self.Linear_estimator2=Linear_estimator23
           self.Linear_estimator3=Linear_estimator23
           
        else:
            
            random_estimator2 = RandomForestRegressor(random_state=0, n_estimators=500)
            random_estimator2.fit(s2,t2)
            self.random_estimator2=random_estimator2
            
            Linear_estimator2 = LinearRegression()
            
            s2r=s2.full_sq
            s2r=s2r.reshape(len(s2r),1)
            t2r=t2
            t2r=t2r.reshape(len(t2r),1)
            Linear_estimator2.fit(s2r, t2r)
            self.Linear_estimator2=Linear_estimator2
            
            
            
            random_estimator3 = RandomForestRegressor(random_state=0, n_estimators=500)
            random_estimator3.fit(s3,t3)
            self.random_estimator3=random_estimator3
            
            Linear_estimator3 = LinearRegression()
            
            s3r=s3.full_sq
            s3r=s3r.reshape(len(s3r),1)
            t3r=t3
            t3r=t3r.reshape(len(t3r),1)
            Linear_estimator3.fit(s3r, t3r)
            self.Linear_estimator3=Linear_estimator3
            
            
            
           
        if self.FLAG2==1:
               self.TwoClassModel=twoclass
        else :
               
                    
                d3=np.ones(len(t3))
                d4=np.ones(len(t4))*-1
                          
                X = pd.concat([s3,s4])
                y = np.concatenate([d3,d4],axis=0)
                self.y=y
                self.X=X
                #print (X)
                #self.decision_clf = svm.SVC()
                TwoClassModel = RandomForestClassifier()
                TwoClassModel.fit(X, y)
                self.TwoClassModel=TwoClassModel
                
            
        Linear_estimator4 = LinearRegression()        
        s4r=s4.full_sq
        s4r=s4r.reshape(len(s4r),1)
        t4r=t4
        t4r=t4r.reshape(len(t4r),1)
        Linear_estimator4.fit(s4r, t4r)
        self.Linear_estimator4=Linear_estimator4             

        
    def predict_price(self,test_row):
        testrow1=test_row
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
        if dum > self.highcutoff1:
            #predicted_price=5
            predicted_price1= self.random_estimator3.predict(test_row)[0] 
            
            x1 = test_row.full_sq
            x1 = x1.reshape(len(x1),1)
            
            predicted_price2= self.Linear_estimator3.predict(x1)[0]
            
        
            predicted_price=0.2*predicted_price1+ 0.8*predicted_price2
            
            
        else:
            if flag==1:
                
                predicted_price1= self.random_estimator2.predict(test_row)[0] 
                x1 = test_row.full_sq
                x1 = x1.reshape(len(x1),1)
                predicted_price2= self.Linear_estimator2.predict(x1)[0]
                predicted_price=0.5*predicted_price1+ 0.5*predicted_price2
            
            else:
                if self.FLAG2==1:
                    testrow1=testrow1.drop(['sub_area', 'full_sq' ,'product_type'],axis=1)
                    testrow1=testrow1.iloc[:,2:20]
                    for fea in testrow1.columns:
                              testrow1[fea].fillna(testrow1[fea].median(),inplace=True)
                    for fea in testrow1.columns:
                              testrow1[fea].fillna(0,inplace=True)
                              
                    pclass=self.TwoClassModel.predict(testrow1)[0]
                
                else:
                       pclass=self.TwoClassModel.predict(test_row)[0]
                       
                if pclass==-1:
                       #predicted_price= 1.5
                       x1 = test_row.full_sq
                       x1 = x1.reshape(len(x1),1)
                       predicted_price= self.Linear_estimator4.predict(x1)[0]
                else:                        
                       predicted_price1= self.random_estimator3.predict(test_row)[0] 
                       x1 = test_row.full_sq
                       x1 = x1.reshape(len(x1),1)
                       predicted_price2= self.Linear_estimator3.predict(x1)[0]
                       predicted_price=.9*predicted_price1+ .1*predicted_price2
                       
        if self.specialFlag==1:
            #if (dum >70) & (dum <600) & (predicted_price > 10):
            #     predicted_price=predicted_price + 3
            if (predicted_price <3) :
                 predicted_price +=dum/50    
        return predicted_price
gc.collect()
###############################################################
class Sviblovo():
    #_data = pd.DataFrame
    def __init__(self, train_rows, twoclass):
        
        #### Flags
        self.specialFlag=0 ### this is for special adjustment in predict
        self.FLAG1=0 ### this is 1 for combined s2 s3
        self.FLAG2=0 ### this is 1 for global TwoClass classifier
        
        
        
        self.highcutoff = 150
        self.highcutoff1=150
        self.lowpricecutoff =3.2

        s1=train_rows[train_rows.full_sq > self.highcutoff];
        sa=train_rows[train_rows.full_sq < self.highcutoff];
        s2=sa[(sa.state.isnull()) & (sa.life_sq.isnull())]
        s3a = sa[~((sa.state.isnull()) & (sa.life_sq.isnull()))]
        s3=s3a[s3a.price_doc>self.lowpricecutoff]
        s4=s3a[s3a.price_doc <= self.lowpricecutoff]
        t1=s1.price_doc
        t2=s2.price_doc
        t3=s3.price_doc
        t4=s4.price_doc
        s1=s1.drop(['price_doc','sub_area','product_type'],axis=1)
        s1=s1.iloc[:,2:20]
        for fea in s1.columns:
                s1[fea].fillna(s1[fea].median(), inplace=True)
        for fea in s1.columns:
                s1[fea].fillna(0, inplace=True)
        s2=s2.drop(['price_doc','sub_area','product_type'],axis=1)
        s2=s2.iloc[:,2:20]
        for fea in s2.columns:
                s2[fea].fillna(s2[fea].median(), inplace=True)
        for fea in s2.columns:
                s2[fea].fillna(0, inplace=True)              
        s3=s3.drop(['price_doc','sub_area','product_type'],axis=1)
        s3=s3.iloc[:,2:20]
        for fea in s3.columns:
                s3[fea].fillna(s3[fea].median(), inplace=True)
        for fea in s3.columns:
                s3[fea].fillna(0, inplace=True)
        s4=s4.drop(['price_doc','sub_area','product_type'],axis=1)
        s4=s4.iloc[:,2:20]
        for fea in s4.columns:
                s4[fea].fillna(s4[fea].median(), inplace=True)
        for fea in s4.columns:
                s4[fea].fillna(0, inplace=True)
        
        sa=sa.drop(['price_doc','sub_area','product_type'],axis=1)
        sa=sa.iloc[:,2:20]
        for fea in sa.columns:
                sa[fea].fillna(sa[fea].median(), inplace=True)
        for fea in sa.columns:
                sa[fea].fillna(0, inplace=True)
                
                
        if self.FLAG1==1:
           X23 = pd.concat([s2,s3]) 
           y23 = np.concatenate([t2,t3],axis=0)
           random_estimator23=RandomForestRegressor(random_state=0, n_estimators=500)
           random_estimator23.fit(X23,y23)
           self.random_estimator2=random_estimator23
           self.random_estimator3=random_estimator23
           
           X23r=X23.full_sq
           X23r=X23r.reshape(len(X23r),1)
           y23r=y23.reshape(len(y23), 1)
           Linear_estimator23=LinearRegression()
           Linear_estimator23.fit(X23r, y23r)
           
           self.Linear_estimator2=Linear_estimator23
           self.Linear_estimator3=Linear_estimator23
           
        else:
            
            random_estimator2 = RandomForestRegressor(random_state=0, n_estimators=500)
            random_estimator2.fit(s2,t2)
            self.random_estimator2=random_estimator2
            
            Linear_estimator2 = LinearRegression()
            
            s2r=s2.full_sq
            s2r=s2r.reshape(len(s2r),1)
            t2r=t2
            t2r=t2r.reshape(len(t2r),1)
            Linear_estimator2.fit(s2r, t2r)
            self.Linear_estimator2=Linear_estimator2
            
            
            
            random_estimator3 = RandomForestRegressor(random_state=0, n_estimators=500)
            random_estimator3.fit(s3,t3)
            self.random_estimator3=random_estimator3
            
            Linear_estimator3 = LinearRegression()
            
            s3r=s3.full_sq
            s3r=s3r.reshape(len(s3r),1)
            t3r=t3
            t3r=t3r.reshape(len(t3r),1)
            Linear_estimator3.fit(s3r, t3r)
            self.Linear_estimator3=Linear_estimator3
            
            
            
           
        if self.FLAG2==1:
               self.TwoClassModel=twoclass
        else :
               
                    
                d3=np.ones(len(t3))
                d4=np.ones(len(t4))*-1
                          
                X = pd.concat([s3,s4])
                y = np.concatenate([d3,d4],axis=0)
                self.y=y
                self.X=X
                #print (X)
                #self.decision_clf = svm.SVC()
                TwoClassModel = RandomForestClassifier()
                TwoClassModel.fit(X, y)
                self.TwoClassModel=TwoClassModel
                
            
        Linear_estimator4 = LinearRegression()        
        s4r=s4.full_sq
        s4r=s4r.reshape(len(s4r),1)
        t4r=t4
        t4r=t4r.reshape(len(t4r),1)
        Linear_estimator4.fit(s4r, t4r)
        self.Linear_estimator4=Linear_estimator4             

        
    def predict_price(self,test_row):
        testrow1=test_row
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
        if dum > self.highcutoff1:
            #predicted_price=5
            predicted_price1= self.random_estimator3.predict(test_row)[0] 
            
            x1 = test_row.full_sq
            x1 = x1.reshape(len(x1),1)
            
            predicted_price2= self.Linear_estimator3.predict(x1)[0]
            
        
            predicted_price=0.2*predicted_price1+ 0.8*predicted_price2
            
            
        else:
            if flag==1:
                
                predicted_price1= self.random_estimator2.predict(test_row)[0] 
                x1 = test_row.full_sq
                x1 = x1.reshape(len(x1),1)
                predicted_price2= self.Linear_estimator2.predict(x1)[0]
                predicted_price=0.5*predicted_price1+ 0.5*predicted_price2
            
            else:
                if self.FLAG2==1:
                    testrow1=testrow1.drop(['sub_area', 'full_sq' ,'product_type'],axis=1)
                    testrow1=testrow1.iloc[:,2:20]
                    for fea in testrow1.columns:
                              testrow1[fea].fillna(testrow1[fea].median(),inplace=True)
                    for fea in testrow1.columns:
                              testrow1[fea].fillna(0,inplace=True)
                              
                    pclass=self.TwoClassModel.predict(testrow1)[0]
                
                else:
                       pclass=self.TwoClassModel.predict(test_row)[0]
                       
                if pclass==-1:
                       #predicted_price= 1.5
                       x1 = test_row.full_sq
                       x1 = x1.reshape(len(x1),1)
                       predicted_price= self.Linear_estimator4.predict(x1)[0]
                else:                        
                       predicted_price1= self.random_estimator3.predict(test_row)[0] 
                       x1 = test_row.full_sq
                       x1 = x1.reshape(len(x1),1)
                       predicted_price2= self.Linear_estimator3.predict(x1)[0]
                       predicted_price=.9*predicted_price1+ .1*predicted_price2
                       
        if self.specialFlag==1:
            #if (dum >70) & (dum <600) & (predicted_price > 10):
            #     predicted_price=predicted_price + 3
            if (predicted_price <3) :
                 predicted_price +=dum/50    
        return predicted_price
gc.collect()
###############################################################

class Silino():
    #_data = pd.DataFrame
    def __init__(self, train_rows, twoclass):
        
        #### Flags
        self.specialFlag=0 ### this is for special adjustment in predict
        self.FLAG1=1 ### this is 1 for combined s2 s3
        self.FLAG2=0 ### this is 1 for global TwoClass classifier
        
        
        
        self.highcutoff = 150
        self.highcutoff1=150
        self.lowpricecutoff =3.2

        s1=train_rows[train_rows.full_sq > self.highcutoff];
        sa=train_rows[train_rows.full_sq < self.highcutoff];
        s2=sa[(sa.state.isnull()) & (sa.life_sq.isnull())]
        s3a = sa[~((sa.state.isnull()) & (sa.life_sq.isnull()))]
        s3=s3a[s3a.price_doc>self.lowpricecutoff]
        s4=s3a[s3a.price_doc <= self.lowpricecutoff]
        t1=s1.price_doc
        t2=s2.price_doc
        t3=s3.price_doc
        t4=s4.price_doc
        s1=s1.drop(['price_doc','sub_area','product_type'],axis=1)
        s1=s1.iloc[:,2:20]
        for fea in s1.columns:
                s1[fea].fillna(s1[fea].median(), inplace=True)
        for fea in s1.columns:
                s1[fea].fillna(0, inplace=True)
        s2=s2.drop(['price_doc','sub_area','product_type'],axis=1)
        s2=s2.iloc[:,2:20]
        for fea in s2.columns:
                s2[fea].fillna(s2[fea].median(), inplace=True)
        for fea in s2.columns:
                s2[fea].fillna(0, inplace=True)              
        s3=s3.drop(['price_doc','sub_area','product_type'],axis=1)
        s3=s3.iloc[:,2:20]
        for fea in s3.columns:
                s3[fea].fillna(s3[fea].median(), inplace=True)
        for fea in s3.columns:
                s3[fea].fillna(0, inplace=True)
        s4=s4.drop(['price_doc','sub_area','product_type'],axis=1)
        s4=s4.iloc[:,2:20]
        for fea in s4.columns:
                s4[fea].fillna(s4[fea].median(), inplace=True)
        for fea in s4.columns:
                s4[fea].fillna(0, inplace=True)
        
        sa=sa.drop(['price_doc','sub_area','product_type'],axis=1)
        sa=sa.iloc[:,2:20]
        for fea in sa.columns:
                sa[fea].fillna(sa[fea].median(), inplace=True)
        for fea in sa.columns:
                sa[fea].fillna(0, inplace=True)
                
                
        if self.FLAG1==1:
           X23 = pd.concat([s2,s3]) 
           y23 = np.concatenate([t2,t3],axis=0)
           random_estimator23=RandomForestRegressor(random_state=0, n_estimators=500)
           random_estimator23.fit(X23,y23)
           self.random_estimator2=random_estimator23
           self.random_estimator3=random_estimator23
           
           X23r=X23.full_sq
           X23r=X23r.reshape(len(X23r),1)
           y23r=y23.reshape(len(y23), 1)
           Linear_estimator23=LinearRegression()
           Linear_estimator23.fit(X23r, y23r)
           
           self.Linear_estimator2=Linear_estimator23
           self.Linear_estimator3=Linear_estimator23
           
        else:
            
            random_estimator2 = RandomForestRegressor(random_state=0, n_estimators=500)
            random_estimator2.fit(s2,t2)
            self.random_estimator2=random_estimator2
            
            Linear_estimator2 = LinearRegression()
            
            s2r=s2.full_sq
            s2r=s2r.reshape(len(s2r),1)
            t2r=t2
            t2r=t2r.reshape(len(t2r),1)
            Linear_estimator2.fit(s2r, t2r)
            self.Linear_estimator2=Linear_estimator2
            
            
            
            random_estimator3 = RandomForestRegressor(random_state=0, n_estimators=500)
            random_estimator3.fit(s3,t3)
            self.random_estimator3=random_estimator3
            
            Linear_estimator3 = LinearRegression()
            
            s3r=s3.full_sq
            s3r=s3r.reshape(len(s3r),1)
            t3r=t3
            t3r=t3r.reshape(len(t3r),1)
            Linear_estimator3.fit(s3r, t3r)
            self.Linear_estimator3=Linear_estimator3
            
            
            
           
        if self.FLAG2==1:
               self.TwoClassModel=twoclass
        else :
               
                    
                d3=np.ones(len(t3))
                d4=np.ones(len(t4))*-1
                          
                X = pd.concat([s3,s4])
                y = np.concatenate([d3,d4],axis=0)
                self.y=y
                self.X=X
                #print (X)
                #self.decision_clf = svm.SVC()
                TwoClassModel = RandomForestClassifier()
                TwoClassModel.fit(X, y)
                self.TwoClassModel=TwoClassModel
                
            
        Linear_estimator4 = LinearRegression()        
        s4r=s4.full_sq
        s4r=s4r.reshape(len(s4r),1)
        t4r=t4
        t4r=t4r.reshape(len(t4r),1)
        Linear_estimator4.fit(s4r, t4r)
        self.Linear_estimator4=Linear_estimator4             

        
    def predict_price(self,test_row):
        testrow1=test_row
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
        if dum > self.highcutoff1:
            #predicted_price=5
            predicted_price1= self.random_estimator3.predict(test_row)[0] 
            
            x1 = test_row.full_sq
            x1 = x1.reshape(len(x1),1)
            
            predicted_price2= self.Linear_estimator3.predict(x1)[0]
            
        
            predicted_price=0.2*predicted_price1+ 0.8*predicted_price2
            
            
        else:
            if flag==1:
                
                predicted_price1= self.random_estimator2.predict(test_row)[0] 
                x1 = test_row.full_sq
                x1 = x1.reshape(len(x1),1)
                predicted_price2= self.Linear_estimator2.predict(x1)[0]
                predicted_price=0.5*predicted_price1+ 0.5*predicted_price2
            
            else:
                if self.FLAG2==1:
                    testrow1=testrow1.drop(['sub_area', 'full_sq' ,'product_type'],axis=1)
                    testrow1=testrow1.iloc[:,2:20]
                    for fea in testrow1.columns:
                              testrow1[fea].fillna(testrow1[fea].median(),inplace=True)
                    for fea in testrow1.columns:
                              testrow1[fea].fillna(0,inplace=True)
                              
                    pclass=self.TwoClassModel.predict(testrow1)[0]
                
                else:
                       pclass=self.TwoClassModel.predict(test_row)[0]
                       
                if pclass==-1:
                       #predicted_price= 1.5
                       x1 = test_row.full_sq
                       x1 = x1.reshape(len(x1),1)
                       predicted_price= self.Linear_estimator4.predict(x1)[0]
                else:                        
                       predicted_price1= self.random_estimator3.predict(test_row)[0] 
                       x1 = test_row.full_sq
                       x1 = x1.reshape(len(x1),1)
                       predicted_price2= self.Linear_estimator3.predict(x1)[0]
                       predicted_price=.9*predicted_price1+ .1*predicted_price2
                       
        if self.specialFlag==1:
            #if (dum >70) & (dum <600) & (predicted_price > 10):
            #     predicted_price=predicted_price + 3
            if (predicted_price <3) :
                 predicted_price +=dum/50    
        return predicted_price
gc.collect()
###############################################################
class Obruchevskoe():
    #_data = pd.DataFrame
    def __init__(self, train_rows, twoclass):
        
        #### Flags
        self.specialFlag=0 ### this is for special adjustment in predict
        self.FLAG1=0 ### this is 1 for combined s2 s3
        self.FLAG2=0 ### this is 1 for global TwoClass classifier
        
        
        
        self.highcutoff = 150
        self.highcutoff1=150
        self.lowpricecutoff =2.2

        s1=train_rows[train_rows.full_sq > self.highcutoff];
        sa=train_rows[train_rows.full_sq < self.highcutoff];
        s2=sa[(sa.state.isnull()) & (sa.life_sq.isnull())]
        s3a = sa[~((sa.state.isnull()) & (sa.life_sq.isnull()))]
        s3=s3a[s3a.price_doc>self.lowpricecutoff]
        s4=s3a[s3a.price_doc <= self.lowpricecutoff]
        t1=s1.price_doc
        t2=s2.price_doc
        t3=s3.price_doc
        t4=s4.price_doc
        s1=s1.drop(['price_doc','sub_area','product_type'],axis=1)
        s1=s1.iloc[:,2:20]
        for fea in s1.columns:
                s1[fea].fillna(s1[fea].median(), inplace=True)
        for fea in s1.columns:
                s1[fea].fillna(0, inplace=True)
        s2=s2.drop(['price_doc','sub_area','product_type'],axis=1)
        s2=s2.iloc[:,2:20]
        for fea in s2.columns:
                s2[fea].fillna(s2[fea].median(), inplace=True)
        for fea in s2.columns:
                s2[fea].fillna(0, inplace=True)              
        s3=s3.drop(['price_doc','sub_area','product_type'],axis=1)
        s3=s3.iloc[:,2:20]
        for fea in s3.columns:
                s3[fea].fillna(s3[fea].median(), inplace=True)
        for fea in s3.columns:
                s3[fea].fillna(0, inplace=True)
        s4=s4.drop(['price_doc','sub_area','product_type'],axis=1)
        s4=s4.iloc[:,2:20]
        for fea in s4.columns:
                s4[fea].fillna(s4[fea].median(), inplace=True)
        for fea in s4.columns:
                s4[fea].fillna(0, inplace=True)
        
        sa=sa.drop(['price_doc','sub_area','product_type'],axis=1)
        sa=sa.iloc[:,2:20]
        for fea in sa.columns:
                sa[fea].fillna(sa[fea].median(), inplace=True)
        for fea in sa.columns:
                sa[fea].fillna(0, inplace=True)
                
                
        if self.FLAG1==1:
           X23 = pd.concat([s2,s3]) 
           y23 = np.concatenate([t2,t3],axis=0)
           random_estimator23=RandomForestRegressor(random_state=0, n_estimators=500)
           random_estimator23.fit(X23,y23)
           self.random_estimator2=random_estimator23
           self.random_estimator3=random_estimator23
           
           X23r=X23.full_sq
           X23r=X23r.reshape(len(X23r),1)
           y23r=y23.reshape(len(y23), 1)
           Linear_estimator23=LinearRegression()
           Linear_estimator23.fit(X23r, y23r)
           
           self.Linear_estimator2=Linear_estimator23
           self.Linear_estimator3=Linear_estimator23
           
        else:
            
            random_estimator2 = RandomForestRegressor(random_state=0, n_estimators=500)
            random_estimator2.fit(s2,t2)
            self.random_estimator2=random_estimator2
            
            Linear_estimator2 = LinearRegression()
            
            s2r=s2.full_sq
            s2r=s2r.reshape(len(s2r),1)
            t2r=t2
            t2r=t2r.reshape(len(t2r),1)
            Linear_estimator2.fit(s2r, t2r)
            self.Linear_estimator2=Linear_estimator2
            
            
            
            random_estimator3 = RandomForestRegressor(random_state=0, n_estimators=500)
            random_estimator3.fit(s3,t3)
            self.random_estimator3=random_estimator3
            
            Linear_estimator3 = LinearRegression()
            
            s3r=s3.full_sq
            s3r=s3r.reshape(len(s3r),1)
            t3r=t3
            t3r=t3r.reshape(len(t3r),1)
            Linear_estimator3.fit(s3r, t3r)
            self.Linear_estimator3=Linear_estimator3
            
            
            
           
        if self.FLAG2==1:
               self.TwoClassModel=twoclass
        else :
               
                    
                d3=np.ones(len(t3))
                d4=np.ones(len(t4))*-1
                          
                X = pd.concat([s3,s4])
                y = np.concatenate([d3,d4],axis=0)
                self.y=y
                self.X=X
                #print (X)
                #self.decision_clf = svm.SVC()
                TwoClassModel = RandomForestClassifier()
                TwoClassModel.fit(X, y)
                self.TwoClassModel=TwoClassModel
                
            
        Linear_estimator4 = LinearRegression()        
        s4r=s4.full_sq
        s4r=s4r.reshape(len(s4r),1)
        t4r=t4
        t4r=t4r.reshape(len(t4r),1)
        Linear_estimator4.fit(s4r, t4r)
        self.Linear_estimator4=Linear_estimator4             

        
    def predict_price(self,test_row):
        testrow1=test_row
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
        if dum > self.highcutoff1:
            #predicted_price=5
            predicted_price1= self.random_estimator3.predict(test_row)[0] 
            
            x1 = test_row.full_sq
            x1 = x1.reshape(len(x1),1)
            
            predicted_price2= self.Linear_estimator3.predict(x1)[0]
            
        
            predicted_price=0.2*predicted_price1+ 0.8*predicted_price2
            
            
        else:
            if flag==1:
                
                predicted_price1= self.random_estimator2.predict(test_row)[0] 
                x1 = test_row.full_sq
                x1 = x1.reshape(len(x1),1)
                predicted_price2= self.Linear_estimator2.predict(x1)[0]
                predicted_price=0.5*predicted_price1+ 0.5*predicted_price2
            
            else:
                if self.FLAG2==1:
                    testrow1=testrow1.drop(['sub_area', 'full_sq' ,'product_type'],axis=1)
                    testrow1=testrow1.iloc[:,2:20]
                    for fea in testrow1.columns:
                              testrow1[fea].fillna(testrow1[fea].median(),inplace=True)
                    for fea in testrow1.columns:
                              testrow1[fea].fillna(0,inplace=True)
                              
                    pclass=self.TwoClassModel.predict(testrow1)[0]
                
                else:
                       pclass=self.TwoClassModel.predict(test_row)[0]
                       
                if pclass==-1:
                       #predicted_price= 1.5
                       x1 = test_row.full_sq
                       x1 = x1.reshape(len(x1),1)
                       predicted_price= self.Linear_estimator4.predict(x1)[0]
                else:                        
                       predicted_price1= self.random_estimator3.predict(test_row)[0] 
                       x1 = test_row.full_sq
                       x1 = x1.reshape(len(x1),1)
                       predicted_price2= self.Linear_estimator3.predict(x1)[0]
                       predicted_price=.9*predicted_price1+ .1*predicted_price2
                       
        if self.specialFlag==1:
            #if (dum >70) & (dum <600) & (predicted_price > 10):
            #     predicted_price=predicted_price + 3
            if (predicted_price <3) :
                 predicted_price +=dum/50    
        return predicted_price



class Butyrskoe():
    #_data = pd.DataFrame
    def __init__(self, train_rows, twoclass):
        
        #### Flags
        self.specialFlag=0 ### this is for special adjustment in predict
        self.FLAG1=1 ### this is 1 for combined s2 s3
        self.FLAG2=0 ### this is 1 for global TwoClass classifier
        
        
        
        self.highcutoff = 150
        self.highcutoff1=150
        self.lowpricecutoff =3.5

        s1=train_rows[train_rows.full_sq > self.highcutoff];
        sa=train_rows[train_rows.full_sq < self.highcutoff];
        s2=sa[(sa.state.isnull()) & (sa.life_sq.isnull())]
        s3a = sa[~((sa.state.isnull()) & (sa.life_sq.isnull()))]
        s3=s3a[s3a.price_doc>self.lowpricecutoff]
        s4=s3a[s3a.price_doc <= self.lowpricecutoff]
        t1=s1.price_doc
        t2=s2.price_doc
        t3=s3.price_doc
        t4=s4.price_doc
        s1=s1.drop(['price_doc','sub_area','product_type'],axis=1)
        s1=s1.iloc[:,2:20]
        for fea in s1.columns:
                s1[fea].fillna(s1[fea].median(), inplace=True)
        for fea in s1.columns:
                s1[fea].fillna(0, inplace=True)
        s2=s2.drop(['price_doc','sub_area','product_type'],axis=1)
        s2=s2.iloc[:,2:20]
        for fea in s2.columns:
                s2[fea].fillna(s2[fea].median(), inplace=True)
        for fea in s2.columns:
                s2[fea].fillna(0, inplace=True)              
        s3=s3.drop(['price_doc','sub_area','product_type'],axis=1)
        s3=s3.iloc[:,2:20]
        for fea in s3.columns:
                s3[fea].fillna(s3[fea].median(), inplace=True)
        for fea in s3.columns:
                s3[fea].fillna(0, inplace=True)
        s4=s4.drop(['price_doc','sub_area','product_type'],axis=1)
        s4=s4.iloc[:,2:20]
        for fea in s4.columns:
                s4[fea].fillna(s4[fea].median(), inplace=True)
        for fea in s4.columns:
                s4[fea].fillna(0, inplace=True)
        
        sa=sa.drop(['price_doc','sub_area','product_type'],axis=1)
        sa=sa.iloc[:,2:20]
        for fea in sa.columns:
                sa[fea].fillna(sa[fea].median(), inplace=True)
        for fea in sa.columns:
                sa[fea].fillna(0, inplace=True)
                
                
        if self.FLAG1==1:
           X23 = pd.concat([s2,s3]) 
           y23 = np.concatenate([t2,t3],axis=0)
           random_estimator23=RandomForestRegressor(random_state=0, n_estimators=500)
           random_estimator23.fit(X23,y23)
           self.random_estimator2=random_estimator23
           self.random_estimator3=random_estimator23
           
           X23r=X23.full_sq
           X23r=X23r.reshape(len(X23r),1)
           y23r=y23.reshape(len(y23), 1)
           Linear_estimator23=LinearRegression()
           Linear_estimator23.fit(X23r, y23r)
           
           self.Linear_estimator2=Linear_estimator23
           self.Linear_estimator3=Linear_estimator23
           
        else:
            
            random_estimator2 = RandomForestRegressor(random_state=0, n_estimators=500)
            random_estimator2.fit(s2,t2)
            self.random_estimator2=random_estimator2
            
            Linear_estimator2 = LinearRegression()
            
            s2r=s2.full_sq
            s2r=s2r.reshape(len(s2r),1)
            t2r=t2
            t2r=t2r.reshape(len(t2r),1)
            Linear_estimator2.fit(s2r, t2r)
            self.Linear_estimator2=Linear_estimator2
            
            
            
            random_estimator3 = RandomForestRegressor(random_state=0, n_estimators=500)
            random_estimator3.fit(s3,t3)
            self.random_estimator3=random_estimator3
            
            Linear_estimator3 = LinearRegression()
            
            s3r=s3.full_sq
            s3r=s3r.reshape(len(s3r),1)
            t3r=t3
            t3r=t3r.reshape(len(t3r),1)
            Linear_estimator3.fit(s3r, t3r)
            self.Linear_estimator3=Linear_estimator3
            
            
            
           
        if self.FLAG2==1:
               self.TwoClassModel=twoclass
        else :
               
                    
                d3=np.ones(len(t3))
                d4=np.ones(len(t4))*-1
                          
                X = pd.concat([s3,s4])
                y = np.concatenate([d3,d4],axis=0)
                self.y=y
                self.X=X
                #print (X)
                #self.decision_clf = svm.SVC()
                TwoClassModel = RandomForestClassifier()
                TwoClassModel.fit(X, y)
                self.TwoClassModel=TwoClassModel
                
            
        Linear_estimator4 = LinearRegression()        
        s4r=s4.full_sq
        s4r=s4r.reshape(len(s4r),1)
        t4r=t4
        t4r=t4r.reshape(len(t4r),1)
        Linear_estimator4.fit(s4r, t4r)
        self.Linear_estimator4=Linear_estimator4             

        
    def predict_price(self,test_row):
        testrow1=test_row
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
        if dum > self.highcutoff1:
            #predicted_price=5
            predicted_price1= self.random_estimator3.predict(test_row)[0] 
            
            x1 = test_row.full_sq
            x1 = x1.reshape(len(x1),1)
            
            predicted_price2= self.Linear_estimator3.predict(x1)[0]
            
        
            predicted_price=0.2*predicted_price1+ 0.8*predicted_price2
            
            
        else:
            if flag==1:
                
                predicted_price1= self.random_estimator2.predict(test_row)[0] 
                x1 = test_row.full_sq
                x1 = x1.reshape(len(x1),1)
                predicted_price2= self.Linear_estimator2.predict(x1)[0]
                predicted_price=0.5*predicted_price1+ 0.5*predicted_price2
            
            else:
                if self.FLAG2==1:
                    testrow1=testrow1.drop(['sub_area', 'full_sq' ,'product_type'],axis=1)
                    testrow1=testrow1.iloc[:,2:20]
                    for fea in testrow1.columns:
                              testrow1[fea].fillna(testrow1[fea].median(),inplace=True)
                    for fea in testrow1.columns:
                              testrow1[fea].fillna(0,inplace=True)
                              
                    pclass=self.TwoClassModel.predict(testrow1)[0]
                
                else:
                       pclass=self.TwoClassModel.predict(test_row)[0]
                       
                if pclass==-1:
                       #predicted_price= 1.5
                       x1 = test_row.full_sq
                       x1 = x1.reshape(len(x1),1)
                       predicted_price= self.Linear_estimator4.predict(x1)[0]
                else:                        
                       predicted_price1= self.random_estimator3.predict(test_row)[0] 
                       x1 = test_row.full_sq
                       x1 = x1.reshape(len(x1),1)
                       predicted_price2= self.Linear_estimator3.predict(x1)[0]
                       predicted_price=.9*predicted_price1+ .1*predicted_price2
                       
        if self.specialFlag==1:
            #if (dum >70) & (dum <600) & (predicted_price > 10):
            #     predicted_price=predicted_price + 3
            if (predicted_price <3) :
                 predicted_price +=dum/50    
        return predicted_price

gc.collect()
###############################################################
class Birjulevo_Vostochnoe():
    #_data = pd.DataFrame
    def __init__(self, train_rows, twoclass):
        
        #### Flags
        self.specialFlag=0 ### this is for special adjustment in predict
        self.FLAG1=1 ### this is 1 for combined s2 s3
        self.FLAG2=0 ### this is 1 for global TwoClass classifier
        
        
        
        self.highcutoff = 150
        self.highcutoff1=150
        self.lowpricecutoff =3.5

        s1=train_rows[train_rows.full_sq > self.highcutoff];
        sa=train_rows[train_rows.full_sq < self.highcutoff];
        s2=sa[(sa.state.isnull()) & (sa.life_sq.isnull())]
        s3a = sa[~((sa.state.isnull()) & (sa.life_sq.isnull()))]
        s3=s3a[s3a.price_doc>self.lowpricecutoff]
        s4=s3a[s3a.price_doc <= self.lowpricecutoff]
        t1=s1.price_doc
        t2=s2.price_doc
        t3=s3.price_doc
        t4=s4.price_doc
        s1=s1.drop(['price_doc','sub_area','product_type'],axis=1)
        s1=s1.iloc[:,2:20]
        for fea in s1.columns:
                s1[fea].fillna(s1[fea].median(), inplace=True)
        for fea in s1.columns:
                s1[fea].fillna(0, inplace=True)
        s2=s2.drop(['price_doc','sub_area','product_type'],axis=1)
        s2=s2.iloc[:,2:20]
        for fea in s2.columns:
                s2[fea].fillna(s2[fea].median(), inplace=True)
        for fea in s2.columns:
                s2[fea].fillna(0, inplace=True)              
        s3=s3.drop(['price_doc','sub_area','product_type'],axis=1)
        s3=s3.iloc[:,2:20]
        for fea in s3.columns:
                s3[fea].fillna(s3[fea].median(), inplace=True)
        for fea in s3.columns:
                s3[fea].fillna(0, inplace=True)
        s4=s4.drop(['price_doc','sub_area','product_type'],axis=1)
        s4=s4.iloc[:,2:20]
        for fea in s4.columns:
                s4[fea].fillna(s4[fea].median(), inplace=True)
        for fea in s4.columns:
                s4[fea].fillna(0, inplace=True)
        
        sa=sa.drop(['price_doc','sub_area','product_type'],axis=1)
        sa=sa.iloc[:,2:20]
        for fea in sa.columns:
                sa[fea].fillna(sa[fea].median(), inplace=True)
        for fea in sa.columns:
                sa[fea].fillna(0, inplace=True)
                
                
        if self.FLAG1==1:
           X23 = pd.concat([s2,s3]) 
           y23 = np.concatenate([t2,t3],axis=0)
           random_estimator23=RandomForestRegressor(random_state=0, n_estimators=500)
           random_estimator23.fit(X23,y23)
           self.random_estimator2=random_estimator23
           self.random_estimator3=random_estimator23
           
           X23r=X23.full_sq
           X23r=X23r.reshape(len(X23r),1)
           y23r=y23.reshape(len(y23), 1)
           Linear_estimator23=LinearRegression()
           Linear_estimator23.fit(X23r, y23r)
           
           self.Linear_estimator2=Linear_estimator23
           self.Linear_estimator3=Linear_estimator23
           
        else:
            
            random_estimator2 = RandomForestRegressor(random_state=0, n_estimators=500)
            random_estimator2.fit(s2,t2)
            self.random_estimator2=random_estimator2
            
            Linear_estimator2 = LinearRegression()
            
            s2r=s2.full_sq
            s2r=s2r.reshape(len(s2r),1)
            t2r=t2
            t2r=t2r.reshape(len(t2r),1)
            Linear_estimator2.fit(s2r, t2r)
            self.Linear_estimator2=Linear_estimator2
            
            
            
            random_estimator3 = RandomForestRegressor(random_state=0, n_estimators=500)
            random_estimator3.fit(s3,t3)
            self.random_estimator3=random_estimator3
            
            Linear_estimator3 = LinearRegression()
            
            s3r=s3.full_sq
            s3r=s3r.reshape(len(s3r),1)
            t3r=t3
            t3r=t3r.reshape(len(t3r),1)
            Linear_estimator3.fit(s3r, t3r)
            self.Linear_estimator3=Linear_estimator3
            
            
            
           
        if self.FLAG2==1:
               self.TwoClassModel=twoclass
        else :
               
                    
                d3=np.ones(len(t3))
                d4=np.ones(len(t4))*-1
                          
                X = pd.concat([s3,s4])
                y = np.concatenate([d3,d4],axis=0)
                self.y=y
                self.X=X
                #print (X)
                #self.decision_clf = svm.SVC()
                TwoClassModel = RandomForestClassifier()
                TwoClassModel.fit(X, y)
                self.TwoClassModel=TwoClassModel
                
            
        Linear_estimator4 = LinearRegression()        
        s4r=s4.full_sq
        s4r=s4r.reshape(len(s4r),1)
        t4r=t4
        t4r=t4r.reshape(len(t4r),1)
        Linear_estimator4.fit(s4r, t4r)
        self.Linear_estimator4=Linear_estimator4             

        
    def predict_price(self,test_row):
        testrow1=test_row
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
        if dum > self.highcutoff1:
            #predicted_price=5
            predicted_price1= self.random_estimator3.predict(test_row)[0] 
            
            x1 = test_row.full_sq
            x1 = x1.reshape(len(x1),1)
            
            predicted_price2= self.Linear_estimator3.predict(x1)[0]
            
        
            predicted_price=0.2*predicted_price1+ 0.8*predicted_price2
            
            
        else:
            if flag==1:
                
                predicted_price1= self.random_estimator2.predict(test_row)[0] 
                x1 = test_row.full_sq
                x1 = x1.reshape(len(x1),1)
                predicted_price2= self.Linear_estimator2.predict(x1)[0]
                predicted_price=0.5*predicted_price1+ 0.5*predicted_price2
            
            else:
                if self.FLAG2==1:
                    testrow1=testrow1.drop(['sub_area', 'full_sq' ,'product_type'],axis=1)
                    testrow1=testrow1.iloc[:,2:20]
                    for fea in testrow1.columns:
                              testrow1[fea].fillna(testrow1[fea].median(),inplace=True)
                    for fea in testrow1.columns:
                              testrow1[fea].fillna(0,inplace=True)
                              
                    pclass=self.TwoClassModel.predict(testrow1)[0]
                
                else:
                       pclass=self.TwoClassModel.predict(test_row)[0]
                       
                if pclass==-1:
                       #predicted_price= 1.5
                       x1 = test_row.full_sq
                       x1 = x1.reshape(len(x1),1)
                       predicted_price= self.Linear_estimator4.predict(x1)[0]
                else:                        
                       predicted_price1= self.random_estimator3.predict(test_row)[0] 
                       x1 = test_row.full_sq
                       x1 = x1.reshape(len(x1),1)
                       predicted_price2= self.Linear_estimator3.predict(x1)[0]
                       predicted_price=.9*predicted_price1+ .1*predicted_price2
                       
        if self.specialFlag==1:
            #if (dum >70) & (dum <600) & (predicted_price > 10):
            #     predicted_price=predicted_price + 3
            if (predicted_price <3) :
                 predicted_price +=dum/50    
        return predicted_price

gc.collect()
###############################################################
class Caricyno():
    #_data = pd.DataFrame
    def __init__(self, train_rows, twoclass):
        
        #### Flags
        self.specialFlag=0 ### this is for special adjustment in predict
        self.FLAG1=1 ### this is 1 for combined s2 s3
        self.FLAG2=0 ### this is 1 for global TwoClass classifier
        
        
        
        self.highcutoff = 150
        self.highcutoff1=150
        self.lowpricecutoff =3.3

        s1=train_rows[train_rows.full_sq > self.highcutoff];
        sa=train_rows[train_rows.full_sq < self.highcutoff];
        s2=sa[(sa.state.isnull()) & (sa.life_sq.isnull())]
        s3a = sa[~((sa.state.isnull()) & (sa.life_sq.isnull()))]
        s3=s3a[s3a.price_doc>self.lowpricecutoff]
        s4=s3a[s3a.price_doc <= self.lowpricecutoff]
        t1=s1.price_doc
        t2=s2.price_doc
        t3=s3.price_doc
        t4=s4.price_doc
        s1=s1.drop(['price_doc','sub_area','product_type'],axis=1)
        s1=s1.iloc[:,2:20]
        for fea in s1.columns:
                s1[fea].fillna(s1[fea].median(), inplace=True)
        for fea in s1.columns:
                s1[fea].fillna(0, inplace=True)
        s2=s2.drop(['price_doc','sub_area','product_type'],axis=1)
        s2=s2.iloc[:,2:20]
        for fea in s2.columns:
                s2[fea].fillna(s2[fea].median(), inplace=True)
        for fea in s2.columns:
                s2[fea].fillna(0, inplace=True)              
        s3=s3.drop(['price_doc','sub_area','product_type'],axis=1)
        s3=s3.iloc[:,2:20]
        for fea in s3.columns:
                s3[fea].fillna(s3[fea].median(), inplace=True)
        for fea in s3.columns:
                s3[fea].fillna(0, inplace=True)
        s4=s4.drop(['price_doc','sub_area','product_type'],axis=1)
        s4=s4.iloc[:,2:20]
        for fea in s4.columns:
                s4[fea].fillna(s4[fea].median(), inplace=True)
        for fea in s4.columns:
                s4[fea].fillna(0, inplace=True)
        
        sa=sa.drop(['price_doc','sub_area','product_type'],axis=1)
        sa=sa.iloc[:,2:20]
        for fea in sa.columns:
                sa[fea].fillna(sa[fea].median(), inplace=True)
        for fea in sa.columns:
                sa[fea].fillna(0, inplace=True)
                
                
        if self.FLAG1==1:
           X23 = pd.concat([s2,s3]) 
           y23 = np.concatenate([t2,t3],axis=0)
           random_estimator23=RandomForestRegressor(random_state=0, n_estimators=500)
           random_estimator23.fit(X23,y23)
           self.random_estimator2=random_estimator23
           self.random_estimator3=random_estimator23
           
           X23r=X23.full_sq
           X23r=X23r.reshape(len(X23r),1)
           y23r=y23.reshape(len(y23), 1)
           Linear_estimator23=LinearRegression()
           Linear_estimator23.fit(X23r, y23r)
           
           self.Linear_estimator2=Linear_estimator23
           self.Linear_estimator3=Linear_estimator23
           
        else:
            
            random_estimator2 = RandomForestRegressor(random_state=0, n_estimators=500)
            random_estimator2.fit(s2,t2)
            self.random_estimator2=random_estimator2
            
            Linear_estimator2 = LinearRegression()
            
            s2r=s2.full_sq
            s2r=s2r.reshape(len(s2r),1)
            t2r=t2
            t2r=t2r.reshape(len(t2r),1)
            Linear_estimator2.fit(s2r, t2r)
            self.Linear_estimator2=Linear_estimator2
            
            
            
            random_estimator3 = RandomForestRegressor(random_state=0, n_estimators=500)
            random_estimator3.fit(s3,t3)
            self.random_estimator3=random_estimator3
            
            Linear_estimator3 = LinearRegression()
            
            s3r=s3.full_sq
            s3r=s3r.reshape(len(s3r),1)
            t3r=t3
            t3r=t3r.reshape(len(t3r),1)
            Linear_estimator3.fit(s3r, t3r)
            self.Linear_estimator3=Linear_estimator3
            
            
            
           
        if self.FLAG2==1:
               self.TwoClassModel=twoclass
        else :
               
                    
                d3=np.ones(len(t3))
                d4=np.ones(len(t4))*-1
                          
                X = pd.concat([s3,s4])
                y = np.concatenate([d3,d4],axis=0)
                self.y=y
                self.X=X
                #print (X)
                #self.decision_clf = svm.SVC()
                TwoClassModel = RandomForestClassifier()
                TwoClassModel.fit(X, y)
                self.TwoClassModel=TwoClassModel
                
            
        Linear_estimator4 = LinearRegression()        
        s4r=s4.full_sq
        s4r=s4r.reshape(len(s4r),1)
        t4r=t4
        t4r=t4r.reshape(len(t4r),1)
        Linear_estimator4.fit(s4r, t4r)
        self.Linear_estimator4=Linear_estimator4             

        
    def predict_price(self,test_row):
        testrow1=test_row
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
        if dum > self.highcutoff1:
            #predicted_price=5
            predicted_price1= self.random_estimator3.predict(test_row)[0] 
            
            x1 = test_row.full_sq
            x1 = x1.reshape(len(x1),1)
            
            predicted_price2= self.Linear_estimator3.predict(x1)[0]
            
        
            predicted_price=0.2*predicted_price1+ 0.8*predicted_price2
            
            
        else:
            if flag==1:
                
                predicted_price1= self.random_estimator2.predict(test_row)[0] 
                x1 = test_row.full_sq
                x1 = x1.reshape(len(x1),1)
                predicted_price2= self.Linear_estimator2.predict(x1)[0]
                predicted_price=0.5*predicted_price1+ 0.5*predicted_price2
            
            else:
                if self.FLAG2==1:
                    testrow1=testrow1.drop(['sub_area', 'full_sq' ,'product_type'],axis=1)
                    testrow1=testrow1.iloc[:,2:20]
                    for fea in testrow1.columns:
                              testrow1[fea].fillna(testrow1[fea].median(),inplace=True)
                    for fea in testrow1.columns:
                              testrow1[fea].fillna(0,inplace=True)
                              
                    pclass=self.TwoClassModel.predict(testrow1)[0]
                
                else:
                       pclass=self.TwoClassModel.predict(test_row)[0]
                       
                if pclass==-1:
                       #predicted_price= 1.5
                       x1 = test_row.full_sq
                       x1 = x1.reshape(len(x1),1)
                       predicted_price= self.Linear_estimator4.predict(x1)[0]
                else:                        
                       predicted_price1= self.random_estimator3.predict(test_row)[0] 
                       x1 = test_row.full_sq
                       x1 = x1.reshape(len(x1),1)
                       predicted_price2= self.Linear_estimator3.predict(x1)[0]
                       predicted_price=.9*predicted_price1+ .1*predicted_price2
                       
        if self.specialFlag==1:
            #if (dum >70) & (dum <600) & (predicted_price > 10):
            #     predicted_price=predicted_price + 3
            if (predicted_price <3) :
                 predicted_price +=dum/50    
        return predicted_price

gc.collect()
###############################################################
class Krylatskoe():
    #_data = pd.DataFrame
    def __init__(self, train_rows, twoclass):
        
        #### Flags
        self.specialFlag=0 ### this is for special adjustment in predict
        self.FLAG1=1 ### this is 1 for combined s2 s3
        self.FLAG2=0 ### this is 1 for global TwoClass classifier
        
        
        
        self.highcutoff = 150
        self.highcutoff1=150
        self.lowpricecutoff =5

        s1=train_rows[train_rows.full_sq > self.highcutoff];
        sa=train_rows[train_rows.full_sq < self.highcutoff];
        s2=sa[(sa.state.isnull()) & (sa.life_sq.isnull())]
        s3a = sa[~((sa.state.isnull()) & (sa.life_sq.isnull()))]
        s3=s3a[s3a.price_doc>self.lowpricecutoff]
        s4=s3a[s3a.price_doc <= self.lowpricecutoff]
        t1=s1.price_doc
        t2=s2.price_doc
        t3=s3.price_doc
        t4=s4.price_doc
        s1=s1.drop(['price_doc','sub_area','product_type'],axis=1)
        s1=s1.iloc[:,2:20]
        for fea in s1.columns:
                s1[fea].fillna(s1[fea].median(), inplace=True)
        for fea in s1.columns:
                s1[fea].fillna(0, inplace=True)
        s2=s2.drop(['price_doc','sub_area','product_type'],axis=1)
        s2=s2.iloc[:,2:20]
        for fea in s2.columns:
                s2[fea].fillna(s2[fea].median(), inplace=True)
        for fea in s2.columns:
                s2[fea].fillna(0, inplace=True)              
        s3=s3.drop(['price_doc','sub_area','product_type'],axis=1)
        s3=s3.iloc[:,2:20]
        for fea in s3.columns:
                s3[fea].fillna(s3[fea].median(), inplace=True)
        for fea in s3.columns:
                s3[fea].fillna(0, inplace=True)
        s4=s4.drop(['price_doc','sub_area','product_type'],axis=1)
        s4=s4.iloc[:,2:20]
        for fea in s4.columns:
                s4[fea].fillna(s4[fea].median(), inplace=True)
        for fea in s4.columns:
                s4[fea].fillna(0, inplace=True)
        
        sa=sa.drop(['price_doc','sub_area','product_type'],axis=1)
        sa=sa.iloc[:,2:20]
        for fea in sa.columns:
                sa[fea].fillna(sa[fea].median(), inplace=True)
        for fea in sa.columns:
                sa[fea].fillna(0, inplace=True)
                
                
        if self.FLAG1==1:
           X23 = pd.concat([s2,s3]) 
           y23 = np.concatenate([t2,t3],axis=0)
           random_estimator23=RandomForestRegressor(random_state=0, n_estimators=500)
           random_estimator23.fit(X23,y23)
           self.random_estimator2=random_estimator23
           self.random_estimator3=random_estimator23
           
           X23r=X23.full_sq
           X23r=X23r.reshape(len(X23r),1)
           y23r=y23.reshape(len(y23), 1)
           Linear_estimator23=LinearRegression()
           Linear_estimator23.fit(X23r, y23r)
           
           self.Linear_estimator2=Linear_estimator23
           self.Linear_estimator3=Linear_estimator23
           
        else:
            
            random_estimator2 = RandomForestRegressor(random_state=0, n_estimators=500)
            random_estimator2.fit(s2,t2)
            self.random_estimator2=random_estimator2
            
            Linear_estimator2 = LinearRegression()
            
            s2r=s2.full_sq
            s2r=s2r.reshape(len(s2r),1)
            t2r=t2
            t2r=t2r.reshape(len(t2r),1)
            Linear_estimator2.fit(s2r, t2r)
            self.Linear_estimator2=Linear_estimator2
            
            
            
            random_estimator3 = RandomForestRegressor(random_state=0, n_estimators=500)
            random_estimator3.fit(s3,t3)
            self.random_estimator3=random_estimator3
            
            Linear_estimator3 = LinearRegression()
            
            s3r=s3.full_sq
            s3r=s3r.reshape(len(s3r),1)
            t3r=t3
            t3r=t3r.reshape(len(t3r),1)
            Linear_estimator3.fit(s3r, t3r)
            self.Linear_estimator3=Linear_estimator3
            
            
            
           
        if self.FLAG2==1:
               self.TwoClassModel=twoclass
        else :
               
                    
                d3=np.ones(len(t3))
                d4=np.ones(len(t4))*-1
                          
                X = pd.concat([s3,s4])
                y = np.concatenate([d3,d4],axis=0)
                self.y=y
                self.X=X
                #print (X)
                #self.decision_clf = svm.SVC()
                TwoClassModel = RandomForestClassifier()
                TwoClassModel.fit(X, y)
                self.TwoClassModel=TwoClassModel
                
            
        Linear_estimator4 = LinearRegression()        
        s4r=s4.full_sq
        s4r=s4r.reshape(len(s4r),1)
        t4r=t4
        t4r=t4r.reshape(len(t4r),1)
        Linear_estimator4.fit(s4r, t4r)
        self.Linear_estimator4=Linear_estimator4             

        
    def predict_price(self,test_row):
        testrow1=test_row
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
        if dum > self.highcutoff1:
            #predicted_price=5
            predicted_price1= self.random_estimator3.predict(test_row)[0] 
            
            x1 = test_row.full_sq
            x1 = x1.reshape(len(x1),1)
            
            predicted_price2= self.Linear_estimator3.predict(x1)[0]
            
        
            predicted_price=0.2*predicted_price1+ 0.8*predicted_price2
            
            
        else:
            if flag==1:
                
                predicted_price1= self.random_estimator2.predict(test_row)[0] 
                x1 = test_row.full_sq
                x1 = x1.reshape(len(x1),1)
                predicted_price2= self.Linear_estimator2.predict(x1)[0]
                predicted_price=0.5*predicted_price1+ 0.5*predicted_price2
            
            else:
                if self.FLAG2==1:
                    testrow1=testrow1.drop(['sub_area', 'full_sq' ,'product_type'],axis=1)
                    testrow1=testrow1.iloc[:,2:20]
                    for fea in testrow1.columns:
                              testrow1[fea].fillna(testrow1[fea].median(),inplace=True)
                    for fea in testrow1.columns:
                              testrow1[fea].fillna(0,inplace=True)
                              
                    pclass=self.TwoClassModel.predict(testrow1)[0]
                
                else:
                       pclass=self.TwoClassModel.predict(test_row)[0]
                       
                if pclass==-1:
                       #predicted_price= 1.5
                       x1 = test_row.full_sq
                       x1 = x1.reshape(len(x1),1)
                       predicted_price= self.Linear_estimator4.predict(x1)[0]
                else:                        
                       predicted_price1= self.random_estimator3.predict(test_row)[0] 
                       x1 = test_row.full_sq
                       x1 = x1.reshape(len(x1),1)
                       predicted_price2= self.Linear_estimator3.predict(x1)[0]
                       predicted_price=.9*predicted_price1+ .1*predicted_price2
                       
        if self.specialFlag==1:
            #if (dum >70) & (dum <600) & (predicted_price > 10):
            #     predicted_price=predicted_price + 3
            if (predicted_price <3) :
                 predicted_price +=dum/50    
        return predicted_price

gc.collect()
###############################################################
class Shhukino():
    #_data = pd.DataFrame
    def __init__(self, train_rows, twoclass):
        
        #### Flags
        self.specialFlag=1 ### this is for special adjustment in predict
        self.FLAG1=1 ### this is 1 for combined s2 s3
        self.FLAG2=0 ### this is 1 for global TwoClass classifier
        
        
        
        self.highcutoff = 150
        self.highcutoff1=150
        self.lowpricecutoff =5

        s1=train_rows[train_rows.full_sq > self.highcutoff];
        sa=train_rows[train_rows.full_sq < self.highcutoff];
        s2=sa[(sa.state.isnull()) & (sa.life_sq.isnull())]
        s3a = sa[~((sa.state.isnull()) & (sa.life_sq.isnull()))]
        s3=s3a[s3a.price_doc>self.lowpricecutoff]
        s4=s3a[s3a.price_doc <= self.lowpricecutoff]
        t1=s1.price_doc
        t2=s2.price_doc
        t3=s3.price_doc
        t4=s4.price_doc
        s1=s1.drop(['price_doc','sub_area','product_type'],axis=1)
        s1=s1.iloc[:,2:20]
        for fea in s1.columns:
                s1[fea].fillna(s1[fea].median(), inplace=True)
        for fea in s1.columns:
                s1[fea].fillna(0, inplace=True)
        s2=s2.drop(['price_doc','sub_area','product_type'],axis=1)
        s2=s2.iloc[:,2:20]
        for fea in s2.columns:
                s2[fea].fillna(s2[fea].median(), inplace=True)
        for fea in s2.columns:
                s2[fea].fillna(0, inplace=True)              
        s3=s3.drop(['price_doc','sub_area','product_type'],axis=1)
        s3=s3.iloc[:,2:20]
        for fea in s3.columns:
                s3[fea].fillna(s3[fea].median(), inplace=True)
        for fea in s3.columns:
                s3[fea].fillna(0, inplace=True)
        s4=s4.drop(['price_doc','sub_area','product_type'],axis=1)
        s4=s4.iloc[:,2:20]
        for fea in s4.columns:
                s4[fea].fillna(s4[fea].median(), inplace=True)
        for fea in s4.columns:
                s4[fea].fillna(0, inplace=True)
        
        sa=sa.drop(['price_doc','sub_area','product_type'],axis=1)
        sa=sa.iloc[:,2:20]
        for fea in sa.columns:
                sa[fea].fillna(sa[fea].median(), inplace=True)
        for fea in sa.columns:
                sa[fea].fillna(0, inplace=True)
                
                
        if self.FLAG1==1:
           X23 = pd.concat([s2,s3]) 
           y23 = np.concatenate([t2,t3],axis=0)
           random_estimator23=RandomForestRegressor(random_state=0, n_estimators=500)
           random_estimator23.fit(X23,y23)
           self.random_estimator2=random_estimator23
           self.random_estimator3=random_estimator23
           
           X23r=X23.full_sq
           X23r=X23r.reshape(len(X23r),1)
           y23r=y23.reshape(len(y23), 1)
           Linear_estimator23=LinearRegression()
           Linear_estimator23.fit(X23r, y23r)
           
           self.Linear_estimator2=Linear_estimator23
           self.Linear_estimator3=Linear_estimator23
           
        else:
            
            random_estimator2 = RandomForestRegressor(random_state=0, n_estimators=500)
            random_estimator2.fit(s2,t2)
            self.random_estimator2=random_estimator2
            
            Linear_estimator2 = LinearRegression()
            
            s2r=s2.full_sq
            s2r=s2r.reshape(len(s2r),1)
            t2r=t2
            t2r=t2r.reshape(len(t2r),1)
            Linear_estimator2.fit(s2r, t2r)
            self.Linear_estimator2=Linear_estimator2
            
            
            
            random_estimator3 = RandomForestRegressor(random_state=0, n_estimators=500)
            random_estimator3.fit(s3,t3)
            self.random_estimator3=random_estimator3
            
            Linear_estimator3 = LinearRegression()
            
            s3r=s3.full_sq
            s3r=s3r.reshape(len(s3r),1)
            t3r=t3
            t3r=t3r.reshape(len(t3r),1)
            Linear_estimator3.fit(s3r, t3r)
            self.Linear_estimator3=Linear_estimator3
            
            
            
           
        if self.FLAG2==1:
               self.TwoClassModel=twoclass
        else :
               
                    
                d3=np.ones(len(t3))
                d4=np.ones(len(t4))*-1
                          
                X = pd.concat([s3,s4])
                y = np.concatenate([d3,d4],axis=0)
                self.y=y
                self.X=X
                #print (X)
                #self.decision_clf = svm.SVC()
                TwoClassModel = RandomForestClassifier()
                TwoClassModel.fit(X, y)
                self.TwoClassModel=TwoClassModel
                
            
        Linear_estimator4 = LinearRegression()        
        s4r=s4.full_sq
        s4r=s4r.reshape(len(s4r),1)
        t4r=t4
        t4r=t4r.reshape(len(t4r),1)
        Linear_estimator4.fit(s4r, t4r)
        self.Linear_estimator4=Linear_estimator4             

        
    def predict_price(self,test_row):
        testrow1=test_row
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
        if dum > self.highcutoff1:
            #predicted_price=5
            predicted_price1= self.random_estimator3.predict(test_row)[0] 
            
            x1 = test_row.full_sq
            x1 = x1.reshape(len(x1),1)
            
            predicted_price2= self.Linear_estimator3.predict(x1)[0]
            
        
            predicted_price=0.2*predicted_price1+ 0.8*predicted_price2
            
            
        else:
            if flag==1:
                
                predicted_price1= self.random_estimator2.predict(test_row)[0] 
                x1 = test_row.full_sq
                x1 = x1.reshape(len(x1),1)
                predicted_price2= self.Linear_estimator2.predict(x1)[0]
                predicted_price=0.5*predicted_price1+ 0.5*predicted_price2
            
            else:
                if self.FLAG2==1:
                    testrow1=testrow1.drop(['sub_area', 'full_sq' ,'product_type'],axis=1)
                    testrow1=testrow1.iloc[:,2:20]
                    for fea in testrow1.columns:
                              testrow1[fea].fillna(testrow1[fea].median(),inplace=True)
                    for fea in testrow1.columns:
                              testrow1[fea].fillna(0,inplace=True)
                              
                    pclass=self.TwoClassModel.predict(testrow1)[0]
                
                else:
                       pclass=self.TwoClassModel.predict(test_row)[0]
                       
                if pclass==-1:
                       #predicted_price= 1.5
                       x1 = test_row.full_sq
                       x1 = x1.reshape(len(x1),1)
                       predicted_price= self.Linear_estimator4.predict(x1)[0]
                else:                        
                       predicted_price1= self.random_estimator3.predict(test_row)[0] 
                       x1 = test_row.full_sq
                       x1 = x1.reshape(len(x1),1)
                       predicted_price2= self.Linear_estimator3.predict(x1)[0]
                       predicted_price=.9*predicted_price1+ .1*predicted_price2
                       
        if self.specialFlag==1:
            #if (dum >70) & (dum <600) & (predicted_price > 10):
            #     predicted_price=predicted_price + 3
            if (predicted_price <3) :
                 predicted_price +=dum/50    
        return predicted_price

gc.collect()
###############################################################
class Tverskoe():
    #_data = pd.DataFrame
    def __init__(self, train_rows, twoclass):
        
        #### Flags
        self.specialFlag=0 ### this is for special adjustment in predict
        self.FLAG1=0  ### this is 1 for combined s2 s3
        self.FLAG2=0 ### this is 1 for global TwoClass classifier
        
        
        
        self.highcutoff = 150
        self.highcutoff1=150
        self.lowpricecutoff =3.2

        s1=train_rows[train_rows.full_sq > self.highcutoff];
        sa=train_rows[train_rows.full_sq < self.highcutoff];
        s2=sa[(sa.state.isnull()) & (sa.life_sq.isnull())]
        s3a = sa[~((sa.state.isnull()) & (sa.life_sq.isnull()))]
        s3=s3a[s3a.price_doc>self.lowpricecutoff]
        s4=s3a[s3a.price_doc <= self.lowpricecutoff]
        t1=s1.price_doc
        t2=s2.price_doc
        t3=s3.price_doc
        t4=s4.price_doc
        s1=s1.drop(['price_doc','sub_area','product_type'],axis=1)
        s1=s1.iloc[:,2:20]
        for fea in s1.columns:
                s1[fea].fillna(s1[fea].median(), inplace=True)
        for fea in s1.columns:
                s1[fea].fillna(0, inplace=True)
        s2=s2.drop(['price_doc','sub_area','product_type'],axis=1)
        s2=s2.iloc[:,2:20]
        for fea in s2.columns:
                s2[fea].fillna(s2[fea].median(), inplace=True)
        for fea in s2.columns:
                s2[fea].fillna(0, inplace=True)              
        s3=s3.drop(['price_doc','sub_area','product_type'],axis=1)
        s3=s3.iloc[:,2:20]
        for fea in s3.columns:
                s3[fea].fillna(s3[fea].median(), inplace=True)
        for fea in s3.columns:
                s3[fea].fillna(0, inplace=True)
        s4=s4.drop(['price_doc','sub_area','product_type'],axis=1)
        s4=s4.iloc[:,2:20]
        for fea in s4.columns:
                s4[fea].fillna(s4[fea].median(), inplace=True)
        for fea in s4.columns:
                s4[fea].fillna(0, inplace=True)
        
        sa=sa.drop(['price_doc','sub_area','product_type'],axis=1)
        sa=sa.iloc[:,2:20]
        for fea in sa.columns:
                sa[fea].fillna(sa[fea].median(), inplace=True)
        for fea in sa.columns:
                sa[fea].fillna(0, inplace=True)
                
                
        if self.FLAG1==1:
           X23 = pd.concat([s2,s3]) 
           y23 = np.concatenate([t2,t3],axis=0)
           random_estimator23=RandomForestRegressor(random_state=0, n_estimators=500)
           random_estimator23.fit(X23,y23)
           self.random_estimator2=random_estimator23
           self.random_estimator3=random_estimator23
           
           X23r=X23.full_sq
           X23r=X23r.reshape(len(X23r),1)
           y23r=y23.reshape(len(y23), 1)
           Linear_estimator23=LinearRegression()
           Linear_estimator23.fit(X23r, y23r)
           
           self.Linear_estimator2=Linear_estimator23
           self.Linear_estimator3=Linear_estimator23
           
        else:
            
            random_estimator2 = RandomForestRegressor(random_state=0, n_estimators=500)
            random_estimator2.fit(s2,t2)
            self.random_estimator2=random_estimator2
            
            Linear_estimator2 = LinearRegression()
            
            s2r=s2.full_sq
            s2r=s2r.reshape(len(s2r),1)
            t2r=t2
            t2r=t2r.reshape(len(t2r),1)
            Linear_estimator2.fit(s2r, t2r)
            self.Linear_estimator2=Linear_estimator2
            
            
            
            random_estimator3 = RandomForestRegressor(random_state=0, n_estimators=500)
            random_estimator3.fit(s3,t3)
            self.random_estimator3=random_estimator3
            
            Linear_estimator3 = LinearRegression()
            
            s3r=s3.full_sq
            s3r=s3r.reshape(len(s3r),1)
            t3r=t3
            t3r=t3r.reshape(len(t3r),1)
            Linear_estimator3.fit(s3r, t3r)
            self.Linear_estimator3=Linear_estimator3
            
            
            
           
        if self.FLAG2==1:
               self.TwoClassModel=twoclass
        else :
               
                    
                d3=np.ones(len(t3))
                d4=np.ones(len(t4))*-1
                          
                X = pd.concat([s3,s4])
                y = np.concatenate([d3,d4],axis=0)
                self.y=y
                self.X=X
                #print (X)
                #self.decision_clf = svm.SVC()
                TwoClassModel = RandomForestClassifier()
                TwoClassModel.fit(X, y)
                self.TwoClassModel=TwoClassModel
                
            
        Linear_estimator4 = LinearRegression()        
        s4r=s4.full_sq
        s4r=s4r.reshape(len(s4r),1)
        t4r=t4
        t4r=t4r.reshape(len(t4r),1)
        Linear_estimator4.fit(s4r, t4r)
        self.Linear_estimator4=Linear_estimator4             

        
    def predict_price(self,test_row):
        testrow1=test_row
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
        if dum > self.highcutoff1:
            #predicted_price=5
            predicted_price1= self.random_estimator3.predict(test_row)[0] 
            
            x1 = test_row.full_sq
            x1 = x1.reshape(len(x1),1)
            
            predicted_price2= self.Linear_estimator3.predict(x1)[0]
            
        
            predicted_price=0.2*predicted_price1+ 0.8*predicted_price2
            
            
        else:
            if flag==1:
                
                predicted_price1= self.random_estimator2.predict(test_row)[0] 
                x1 = test_row.full_sq
                x1 = x1.reshape(len(x1),1)
                predicted_price2= self.Linear_estimator2.predict(x1)[0]
                predicted_price=0.5*predicted_price1+ 0.5*predicted_price2
            
            else:
                if self.FLAG2==1:
                    testrow1=testrow1.drop(['sub_area', 'full_sq' ,'product_type'],axis=1)
                    testrow1=testrow1.iloc[:,2:20]
                    for fea in testrow1.columns:
                              testrow1[fea].fillna(testrow1[fea].median(),inplace=True)
                    for fea in testrow1.columns:
                              testrow1[fea].fillna(0,inplace=True)
                              
                    pclass=self.TwoClassModel.predict(testrow1)[0]
                
                else:
                       pclass=self.TwoClassModel.predict(test_row)[0]
                       
                if pclass==-1:
                       #predicted_price= 1.5
                       x1 = test_row.full_sq
                       x1 = x1.reshape(len(x1),1)
                       predicted_price= self.Linear_estimator4.predict(x1)[0]
                else:                        
                       predicted_price1= self.random_estimator3.predict(test_row)[0] 
                       x1 = test_row.full_sq
                       x1 = x1.reshape(len(x1),1)
                       predicted_price2= self.Linear_estimator3.predict(x1)[0]
                       predicted_price=.9*predicted_price1+ .1*predicted_price2
                       
        if self.specialFlag==1:
            #if (dum >70) & (dum <600) & (predicted_price > 10):
            #     predicted_price=predicted_price + 3
            if (predicted_price <3) :
                 predicted_price +=dum/50    
        return predicted_price

gc.collect()
###############################################################
class Taganskoe():
    #_data = pd.DataFrame
    def __init__(self, train_rows, twoclass):
        
        #### Flags
        self.specialFlag=0 ### this is for special adjustment in predict
        self.FLAG1=1  ### this is 1 for combined s2 s3
        self.FLAG2=0 ### this is 1 for global TwoClass classifier
        
        
        
        self.highcutoff = 150
        self.highcutoff1=150
        self.lowpricecutoff =3.2

        s1=train_rows[train_rows.full_sq > self.highcutoff];
        sa=train_rows[train_rows.full_sq < self.highcutoff];
        s2=sa[(sa.state.isnull()) & (sa.life_sq.isnull())]
        s3a = sa[~((sa.state.isnull()) & (sa.life_sq.isnull()))]
        s3=s3a[s3a.price_doc>self.lowpricecutoff]
        s4=s3a[s3a.price_doc <= self.lowpricecutoff]
        t1=s1.price_doc
        t2=s2.price_doc
        t3=s3.price_doc
        t4=s4.price_doc
        s1=s1.drop(['price_doc','sub_area','product_type'],axis=1)
        s1=s1.iloc[:,2:20]
        for fea in s1.columns:
                s1[fea].fillna(s1[fea].median(), inplace=True)
        for fea in s1.columns:
                s1[fea].fillna(0, inplace=True)
        s2=s2.drop(['price_doc','sub_area','product_type'],axis=1)
        s2=s2.iloc[:,2:20]
        for fea in s2.columns:
                s2[fea].fillna(s2[fea].median(), inplace=True)
        for fea in s2.columns:
                s2[fea].fillna(0, inplace=True)              
        s3=s3.drop(['price_doc','sub_area','product_type'],axis=1)
        s3=s3.iloc[:,2:20]
        for fea in s3.columns:
                s3[fea].fillna(s3[fea].median(), inplace=True)
        for fea in s3.columns:
                s3[fea].fillna(0, inplace=True)
        s4=s4.drop(['price_doc','sub_area','product_type'],axis=1)
        s4=s4.iloc[:,2:20]
        for fea in s4.columns:
                s4[fea].fillna(s4[fea].median(), inplace=True)
        for fea in s4.columns:
                s4[fea].fillna(0, inplace=True)
        
        sa=sa.drop(['price_doc','sub_area','product_type'],axis=1)
        sa=sa.iloc[:,2:20]
        for fea in sa.columns:
                sa[fea].fillna(sa[fea].median(), inplace=True)
        for fea in sa.columns:
                sa[fea].fillna(0, inplace=True)
                
                
        if self.FLAG1==1:
           X23 = pd.concat([s2,s3]) 
           y23 = np.concatenate([t2,t3],axis=0)
           random_estimator23=RandomForestRegressor(random_state=0, n_estimators=500)
           random_estimator23.fit(X23,y23)
           self.random_estimator2=random_estimator23
           self.random_estimator3=random_estimator23
           
           X23r=X23.full_sq
           X23r=X23r.reshape(len(X23r),1)
           y23r=y23.reshape(len(y23), 1)
           Linear_estimator23=LinearRegression()
           Linear_estimator23.fit(X23r, y23r)
           
           self.Linear_estimator2=Linear_estimator23
           self.Linear_estimator3=Linear_estimator23
           
        else:
            
            random_estimator2 = RandomForestRegressor(random_state=0, n_estimators=500)
            random_estimator2.fit(s2,t2)
            self.random_estimator2=random_estimator2
            
            Linear_estimator2 = LinearRegression()
            
            s2r=s2.full_sq
            s2r=s2r.reshape(len(s2r),1)
            t2r=t2
            t2r=t2r.reshape(len(t2r),1)
            Linear_estimator2.fit(s2r, t2r)
            self.Linear_estimator2=Linear_estimator2
            
            
            
            random_estimator3 = RandomForestRegressor(random_state=0, n_estimators=500)
            random_estimator3.fit(s3,t3)
            self.random_estimator3=random_estimator3
            
            Linear_estimator3 = LinearRegression()
            
            s3r=s3.full_sq
            s3r=s3r.reshape(len(s3r),1)
            t3r=t3
            t3r=t3r.reshape(len(t3r),1)
            Linear_estimator3.fit(s3r, t3r)
            self.Linear_estimator3=Linear_estimator3
            
            
            
           
        if self.FLAG2==1:
               self.TwoClassModel=twoclass
        else :
               
                    
                d3=np.ones(len(t3))
                d4=np.ones(len(t4))*-1
                          
                X = pd.concat([s3,s4])
                y = np.concatenate([d3,d4],axis=0)
                self.y=y
                self.X=X
                #print (X)
                #self.decision_clf = svm.SVC()
                TwoClassModel = RandomForestClassifier()
                TwoClassModel.fit(X, y)
                self.TwoClassModel=TwoClassModel
                
            
        Linear_estimator4 = LinearRegression()        
        s4r=s4.full_sq
        s4r=s4r.reshape(len(s4r),1)
        t4r=t4
        t4r=t4r.reshape(len(t4r),1)
        Linear_estimator4.fit(s4r, t4r)
        self.Linear_estimator4=Linear_estimator4             

        
    def predict_price(self,test_row):
        testrow1=test_row
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
        if dum > self.highcutoff1:
            #predicted_price=5
            predicted_price1= self.random_estimator3.predict(test_row)[0] 
            
            x1 = test_row.full_sq
            x1 = x1.reshape(len(x1),1)
            
            predicted_price2= self.Linear_estimator3.predict(x1)[0]
            
        
            predicted_price=0.2*predicted_price1+ 0.8*predicted_price2
            
            
        else:
            if flag==1:
                
                predicted_price1= self.random_estimator2.predict(test_row)[0] 
                x1 = test_row.full_sq
                x1 = x1.reshape(len(x1),1)
                predicted_price2= self.Linear_estimator2.predict(x1)[0]
                predicted_price=0.5*predicted_price1+ 0.5*predicted_price2
            
            else:
                if self.FLAG2==1:
                    testrow1=testrow1.drop(['sub_area', 'full_sq' ,'product_type'],axis=1)
                    testrow1=testrow1.iloc[:,2:20]
                    for fea in testrow1.columns:
                              testrow1[fea].fillna(testrow1[fea].median(),inplace=True)
                    for fea in testrow1.columns:
                              testrow1[fea].fillna(0,inplace=True)
                              
                    pclass=self.TwoClassModel.predict(testrow1)[0]
                
                else:
                       pclass=self.TwoClassModel.predict(test_row)[0]
                       
                if pclass==-1:
                       #predicted_price= 1.5
                       x1 = test_row.full_sq
                       x1 = x1.reshape(len(x1),1)
                       predicted_price= self.Linear_estimator4.predict(x1)[0]
                else:                        
                       predicted_price1= self.random_estimator3.predict(test_row)[0] 
                       x1 = test_row.full_sq
                       x1 = x1.reshape(len(x1),1)
                       predicted_price2= self.Linear_estimator3.predict(x1)[0]
                       predicted_price=.9*predicted_price1+ .1*predicted_price2
                       
        if self.specialFlag==1:
            #if (dum >70) & (dum <600) & (predicted_price > 10):
            #     predicted_price=predicted_price + 3
            if (predicted_price <3) :
                 predicted_price +=dum/50    
        return predicted_price

gc.collect()
###############################################################
class Kapotnja():
    #_data = pd.DataFrame
    def __init__(self, train_rows, twoclass):
        
        #### Flags
        self.specialFlag=0 ### this is for special adjustment in predict
        self.FLAG1=0  ### this is 1 for combined s2 s3
        self.FLAG2=0 ### this is 1 for global TwoClass classifier
        
        
        
        self.highcutoff = 150
        self.highcutoff1=150
        self.lowpricecutoff =2

        s1=train_rows[train_rows.full_sq > self.highcutoff];
        sa=train_rows[train_rows.full_sq < self.highcutoff];
        s2=sa[(sa.state.isnull()) & (sa.life_sq.isnull())]
        s3a = sa[~((sa.state.isnull()) & (sa.life_sq.isnull()))]
        s3=s3a[s3a.price_doc>self.lowpricecutoff]
        s4=s3a[s3a.price_doc <= self.lowpricecutoff]
        t1=s1.price_doc
        t2=s2.price_doc
        t3=s3.price_doc
        t4=s4.price_doc
        s1=s1.drop(['price_doc','sub_area','product_type'],axis=1)
        s1=s1.iloc[:,2:20]
        for fea in s1.columns:
                s1[fea].fillna(s1[fea].median(), inplace=True)
        for fea in s1.columns:
                s1[fea].fillna(0, inplace=True)
        s2=s2.drop(['price_doc','sub_area','product_type'],axis=1)
        s2=s2.iloc[:,2:20]
        for fea in s2.columns:
                s2[fea].fillna(s2[fea].median(), inplace=True)
        for fea in s2.columns:
                s2[fea].fillna(0, inplace=True)              
        s3=s3.drop(['price_doc','sub_area','product_type'],axis=1)
        s3=s3.iloc[:,2:20]
        for fea in s3.columns:
                s3[fea].fillna(s3[fea].median(), inplace=True)
        for fea in s3.columns:
                s3[fea].fillna(0, inplace=True)
        s4=s4.drop(['price_doc','sub_area','product_type'],axis=1)
        s4=s4.iloc[:,2:20]
        for fea in s4.columns:
                s4[fea].fillna(s4[fea].median(), inplace=True)
        for fea in s4.columns:
                s4[fea].fillna(0, inplace=True)
        
        sa=sa.drop(['price_doc','sub_area','product_type'],axis=1)
        sa=sa.iloc[:,2:20]
        for fea in sa.columns:
                sa[fea].fillna(sa[fea].median(), inplace=True)
        for fea in sa.columns:
                sa[fea].fillna(0, inplace=True)
                
                
        if self.FLAG1==1:
           X23 = pd.concat([s2,s3]) 
           y23 = np.concatenate([t2,t3],axis=0)
           random_estimator23=RandomForestRegressor(random_state=0, n_estimators=500)
           random_estimator23.fit(X23,y23)
           self.random_estimator2=random_estimator23
           self.random_estimator3=random_estimator23
           
           X23r=X23.full_sq
           X23r=X23r.reshape(len(X23r),1)
           y23r=y23.reshape(len(y23), 1)
           Linear_estimator23=LinearRegression()
           Linear_estimator23.fit(X23r, y23r)
           
           self.Linear_estimator2=Linear_estimator23
           self.Linear_estimator3=Linear_estimator23
           
        else:
            
            random_estimator2 = RandomForestRegressor(random_state=0, n_estimators=500)
            random_estimator2.fit(s2,t2)
            self.random_estimator2=random_estimator2
            
            Linear_estimator2 = LinearRegression()
            
            s2r=s2.full_sq
            s2r=s2r.reshape(len(s2r),1)
            t2r=t2
            t2r=t2r.reshape(len(t2r),1)
            Linear_estimator2.fit(s2r, t2r)
            self.Linear_estimator2=Linear_estimator2
            
            
            
            random_estimator3 = RandomForestRegressor(random_state=0, n_estimators=500)
            random_estimator3.fit(s3,t3)
            self.random_estimator3=random_estimator3
            
            Linear_estimator3 = LinearRegression()
            
            s3r=s3.full_sq
            s3r=s3r.reshape(len(s3r),1)
            t3r=t3
            t3r=t3r.reshape(len(t3r),1)
            Linear_estimator3.fit(s3r, t3r)
            self.Linear_estimator3=Linear_estimator3
            
            
            
           
        if self.FLAG2==1:
               self.TwoClassModel=twoclass
        else :
               
                    
                d3=np.ones(len(t3))
                d4=np.ones(len(t4))*-1
                          
                X = pd.concat([s3,s4])
                y = np.concatenate([d3,d4],axis=0)
                self.y=y
                self.X=X
                #print (X)
                #self.decision_clf = svm.SVC()
                TwoClassModel = RandomForestClassifier()
                TwoClassModel.fit(X, y)
                self.TwoClassModel=TwoClassModel
                
            
        Linear_estimator4 = LinearRegression()        
        s4r=s4.full_sq
        s4r=s4r.reshape(len(s4r),1)
        t4r=t4
        t4r=t4r.reshape(len(t4r),1)
        Linear_estimator4.fit(s4r, t4r)
        self.Linear_estimator4=Linear_estimator4             

        
    def predict_price(self,test_row):
        testrow1=test_row
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
        if dum > self.highcutoff1:
            #predicted_price=5
            predicted_price1= self.random_estimator3.predict(test_row)[0] 
            
            x1 = test_row.full_sq
            x1 = x1.reshape(len(x1),1)
            
            predicted_price2= self.Linear_estimator3.predict(x1)[0]
            
        
            predicted_price=0.2*predicted_price1+ 0.8*predicted_price2
            
            
        else:
            if flag==1:
                
                predicted_price1= self.random_estimator2.predict(test_row)[0] 
                x1 = test_row.full_sq
                x1 = x1.reshape(len(x1),1)
                predicted_price2= self.Linear_estimator2.predict(x1)[0]
                predicted_price=0.5*predicted_price1+ 0.5*predicted_price2
            
            else:
                if self.FLAG2==1:
                    testrow1=testrow1.drop(['sub_area', 'full_sq' ,'product_type'],axis=1)
                    testrow1=testrow1.iloc[:,2:20]
                    for fea in testrow1.columns:
                              testrow1[fea].fillna(testrow1[fea].median(),inplace=True)
                    for fea in testrow1.columns:
                              testrow1[fea].fillna(0,inplace=True)
                              
                    pclass=self.TwoClassModel.predict(testrow1)[0]
                
                else:
                       pclass=self.TwoClassModel.predict(test_row)[0]
                       
                if pclass==-1:
                       #predicted_price= 1.5
                       x1 = test_row.full_sq
                       x1 = x1.reshape(len(x1),1)
                       predicted_price= self.Linear_estimator4.predict(x1)[0]
                else:                        
                       predicted_price1= self.random_estimator3.predict(test_row)[0] 
                       x1 = test_row.full_sq
                       x1 = x1.reshape(len(x1),1)
                       predicted_price2= self.Linear_estimator3.predict(x1)[0]
                       predicted_price=.9*predicted_price1+ .1*predicted_price2
                       
        if self.specialFlag==1:
            #if (dum >70) & (dum <600) & (predicted_price > 10):
            #     predicted_price=predicted_price + 3
            if (predicted_price <3) :
                 predicted_price +=dum/50
        predicted_price -= .1
        return predicted_price 
gc.collect()
###############################################################
class Horoshevskoe():
    #_data = pd.DataFrame
    def __init__(self, train_rows, twoclass):
        
        #### Flags
        self.specialFlag=0 ### this is for special adjustment in predict
        self.FLAG1=0  ### this is 1 for combined s2 s3
        self.FLAG2=0 ### this is 1 for global TwoClass classifier
        
        
        
        self.highcutoff = 150
        self.highcutoff1=150
        self.lowpricecutoff =3.2

        s1=train_rows[train_rows.full_sq > self.highcutoff];
        sa=train_rows[train_rows.full_sq < self.highcutoff];
        s2=sa[(sa.state.isnull()) & (sa.life_sq.isnull())]
        s3a = sa[~((sa.state.isnull()) & (sa.life_sq.isnull()))]
        s3=s3a[s3a.price_doc>self.lowpricecutoff]
        s4=s3a[s3a.price_doc <= self.lowpricecutoff]
        t1=s1.price_doc
        t2=s2.price_doc
        t3=s3.price_doc
        t4=s4.price_doc
        s1=s1.drop(['price_doc','sub_area','product_type'],axis=1)
        s1=s1.iloc[:,2:20]
        for fea in s1.columns:
                s1[fea].fillna(s1[fea].median(), inplace=True)
        for fea in s1.columns:
                s1[fea].fillna(0, inplace=True)
        s2=s2.drop(['price_doc','sub_area','product_type'],axis=1)
        s2=s2.iloc[:,2:20]
        for fea in s2.columns:
                s2[fea].fillna(s2[fea].median(), inplace=True)
        for fea in s2.columns:
                s2[fea].fillna(0, inplace=True)              
        s3=s3.drop(['price_doc','sub_area','product_type'],axis=1)
        s3=s3.iloc[:,2:20]
        for fea in s3.columns:
                s3[fea].fillna(s3[fea].median(), inplace=True)
        for fea in s3.columns:
                s3[fea].fillna(0, inplace=True)
        s4=s4.drop(['price_doc','sub_area','product_type'],axis=1)
        s4=s4.iloc[:,2:20]
        for fea in s4.columns:
                s4[fea].fillna(s4[fea].median(), inplace=True)
        for fea in s4.columns:
                s4[fea].fillna(0, inplace=True)
        
        sa=sa.drop(['price_doc','sub_area','product_type'],axis=1)
        sa=sa.iloc[:,2:20]
        for fea in sa.columns:
                sa[fea].fillna(sa[fea].median(), inplace=True)
        for fea in sa.columns:
                sa[fea].fillna(0, inplace=True)
                
                
        if self.FLAG1==1:
           X23 = pd.concat([s2,s3]) 
           y23 = np.concatenate([t2,t3],axis=0)
           random_estimator23=RandomForestRegressor(random_state=0, n_estimators=500)
           random_estimator23.fit(X23,y23)
           self.random_estimator2=random_estimator23
           self.random_estimator3=random_estimator23
           
           X23r=X23.full_sq
           X23r=X23r.reshape(len(X23r),1)
           y23r=y23.reshape(len(y23), 1)
           Linear_estimator23=LinearRegression()
           Linear_estimator23.fit(X23r, y23r)
           
           self.Linear_estimator2=Linear_estimator23
           self.Linear_estimator3=Linear_estimator23
           
        else:
            
            random_estimator2 = RandomForestRegressor(random_state=0, n_estimators=500)
            random_estimator2.fit(s2,t2)
            self.random_estimator2=random_estimator2
            
            Linear_estimator2 = LinearRegression()
            
            s2r=s2.full_sq
            s2r=s2r.reshape(len(s2r),1)
            t2r=t2
            t2r=t2r.reshape(len(t2r),1)
            Linear_estimator2.fit(s2r, t2r)
            self.Linear_estimator2=Linear_estimator2
            
            
            
            random_estimator3 = RandomForestRegressor(random_state=0, n_estimators=500)
            random_estimator3.fit(s3,t3)
            self.random_estimator3=random_estimator3
            
            Linear_estimator3 = LinearRegression()
            
            s3r=s3.full_sq
            s3r=s3r.reshape(len(s3r),1)
            t3r=t3
            t3r=t3r.reshape(len(t3r),1)
            Linear_estimator3.fit(s3r, t3r)
            self.Linear_estimator3=Linear_estimator3
            
            
            
           
        if self.FLAG2==1:
               self.TwoClassModel=twoclass
        else :
               
                    
                d3=np.ones(len(t3))
                d4=np.ones(len(t4))*-1
                          
                X = pd.concat([s3,s4])
                y = np.concatenate([d3,d4],axis=0)
                self.y=y
                self.X=X
                #print (X)
                #self.decision_clf = svm.SVC()
                TwoClassModel = RandomForestClassifier()
                TwoClassModel.fit(X, y)
                self.TwoClassModel=TwoClassModel
                
            
        Linear_estimator4 = LinearRegression()        
        s4r=s4.full_sq
        s4r=s4r.reshape(len(s4r),1)
        t4r=t4
        t4r=t4r.reshape(len(t4r),1)
        Linear_estimator4.fit(s4r, t4r)
        self.Linear_estimator4=Linear_estimator4             

        
    def predict_price(self,test_row):
        testrow1=test_row
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
        if dum > self.highcutoff1:
            #predicted_price=5
            predicted_price1= self.random_estimator3.predict(test_row)[0] 
            
            x1 = test_row.full_sq
            x1 = x1.reshape(len(x1),1)
            
            predicted_price2= self.Linear_estimator3.predict(x1)[0]
            
        
            predicted_price=0.2*predicted_price1+ 0.8*predicted_price2
            
            
        else:
            if flag==1:
                
                predicted_price1= self.random_estimator2.predict(test_row)[0] 
                x1 = test_row.full_sq
                x1 = x1.reshape(len(x1),1)
                predicted_price2= self.Linear_estimator2.predict(x1)[0]
                predicted_price=0.5*predicted_price1+ 0.5*predicted_price2
            
            else:
                if self.FLAG2==1:
                    testrow1=testrow1.drop(['sub_area', 'full_sq' ,'product_type'],axis=1)
                    testrow1=testrow1.iloc[:,2:20]
                    for fea in testrow1.columns:
                              testrow1[fea].fillna(testrow1[fea].median(),inplace=True)
                    for fea in testrow1.columns:
                              testrow1[fea].fillna(0,inplace=True)
                              
                    pclass=self.TwoClassModel.predict(testrow1)[0]
                
                else:
                       pclass=self.TwoClassModel.predict(test_row)[0]
                       
                if pclass==-1:
                       #predicted_price= 1.5
                       x1 = test_row.full_sq
                       x1 = x1.reshape(len(x1),1)
                       predicted_price= self.Linear_estimator4.predict(x1)[0]
                else:                        
                       predicted_price1= self.random_estimator3.predict(test_row)[0] 
                       x1 = test_row.full_sq
                       x1 = x1.reshape(len(x1),1)
                       predicted_price2= self.Linear_estimator3.predict(x1)[0]
                       predicted_price=.9*predicted_price1+ .1*predicted_price2
                       
        if self.specialFlag==1:
            #if (dum >70) & (dum <600) & (predicted_price > 10):
            #     predicted_price=predicted_price + 3
            if (predicted_price <3) :
                 predicted_price +=dum/50
        #predicted_price -= .1
        return predicted_price 

gc.collect()
###############################################################
class Orehovo_Borisovo_Juzhnoe():
    #_data = pd.DataFrame
    def __init__(self, train_rows, twoclass):
        
        #### Flags
        self.specialFlag=0 ### this is for special adjustment in predict
        self.FLAG1=1  ### this is 1 for combined s2 s3
        self.FLAG2=1 ### this is 1 for global TwoClass classifier
        
        
        
        self.highcutoff = 150
        self.highcutoff1=150
        self.lowpricecutoff =4

        s1=train_rows[train_rows.full_sq > self.highcutoff];
        sa=train_rows[train_rows.full_sq < self.highcutoff];
        s2=sa[(sa.state.isnull()) & (sa.life_sq.isnull())]
        s3a = sa[~((sa.state.isnull()) & (sa.life_sq.isnull()))]
        s3=s3a[s3a.price_doc>self.lowpricecutoff]
        s4=s3a[s3a.price_doc <= self.lowpricecutoff]
        t1=s1.price_doc
        t2=s2.price_doc
        t3=s3.price_doc
        t4=s4.price_doc
        s1=s1.drop(['price_doc','sub_area','product_type'],axis=1)
        s1=s1.iloc[:,2:20]
        for fea in s1.columns:
                s1[fea].fillna(s1[fea].median(), inplace=True)
        for fea in s1.columns:
                s1[fea].fillna(0, inplace=True)
        s2=s2.drop(['price_doc','sub_area','product_type'],axis=1)
        s2=s2.iloc[:,2:20]
        for fea in s2.columns:
                s2[fea].fillna(s2[fea].median(), inplace=True)
        for fea in s2.columns:
                s2[fea].fillna(0, inplace=True)              
        s3=s3.drop(['price_doc','sub_area','product_type'],axis=1)
        s3=s3.iloc[:,2:20]
        for fea in s3.columns:
                s3[fea].fillna(s3[fea].median(), inplace=True)
        for fea in s3.columns:
                s3[fea].fillna(0, inplace=True)
        s4=s4.drop(['price_doc','sub_area','product_type'],axis=1)
        s4=s4.iloc[:,2:20]
        for fea in s4.columns:
                s4[fea].fillna(s4[fea].median(), inplace=True)
        for fea in s4.columns:
                s4[fea].fillna(0, inplace=True)
        
        sa=sa.drop(['price_doc','sub_area','product_type'],axis=1)
        sa=sa.iloc[:,2:20]
        for fea in sa.columns:
                sa[fea].fillna(sa[fea].median(), inplace=True)
        for fea in sa.columns:
                sa[fea].fillna(0, inplace=True)
                
                
        if self.FLAG1==1:
           X23 = pd.concat([s2,s3]) 
           y23 = np.concatenate([t2,t3],axis=0)
           random_estimator23=RandomForestRegressor(random_state=0, n_estimators=500)
           random_estimator23.fit(X23,y23)
           self.random_estimator2=random_estimator23
           self.random_estimator3=random_estimator23
           
           X23r=X23.full_sq
           X23r=X23r.reshape(len(X23r),1)
           y23r=y23.reshape(len(y23), 1)
           Linear_estimator23=LinearRegression()
           Linear_estimator23.fit(X23r, y23r)
           
           self.Linear_estimator2=Linear_estimator23
           self.Linear_estimator3=Linear_estimator23
           
        else:
            
            random_estimator2 = RandomForestRegressor(random_state=0, n_estimators=500)
            random_estimator2.fit(s2,t2)
            self.random_estimator2=random_estimator2
            
            Linear_estimator2 = LinearRegression()
            
            s2r=s2.full_sq
            s2r=s2r.reshape(len(s2r),1)
            t2r=t2
            t2r=t2r.reshape(len(t2r),1)
            Linear_estimator2.fit(s2r, t2r)
            self.Linear_estimator2=Linear_estimator2
            
            
            
            random_estimator3 = RandomForestRegressor(random_state=0, n_estimators=500)
            random_estimator3.fit(s3,t3)
            self.random_estimator3=random_estimator3
            
            Linear_estimator3 = LinearRegression()
            
            s3r=s3.full_sq
            s3r=s3r.reshape(len(s3r),1)
            t3r=t3
            t3r=t3r.reshape(len(t3r),1)
            Linear_estimator3.fit(s3r, t3r)
            self.Linear_estimator3=Linear_estimator3
            
            
            
           
        if self.FLAG2==1:
               self.TwoClassModel=twoclass
        else :
               
                    
                d3=np.ones(len(t3))
                d4=np.ones(len(t4))*-1
                          
                X = pd.concat([s3,s4])
                y = np.concatenate([d3,d4],axis=0)
                self.y=y
                self.X=X
                #print (X)
                #self.decision_clf = svm.SVC()
                TwoClassModel = RandomForestClassifier()
                TwoClassModel.fit(X, y)
                self.TwoClassModel=TwoClassModel
                
            
        Linear_estimator4 = LinearRegression()        
        s4r=s4.full_sq
        s4r=s4r.reshape(len(s4r),1)
        t4r=t4
        t4r=t4r.reshape(len(t4r),1)
        Linear_estimator4.fit(s4r, t4r)
        self.Linear_estimator4=Linear_estimator4             

        
    def predict_price(self,test_row):
        testrow1=test_row
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
        if dum > self.highcutoff1:
            #predicted_price=5
            predicted_price1= self.random_estimator3.predict(test_row)[0] 
            
            x1 = test_row.full_sq
            x1 = x1.reshape(len(x1),1)
            
            predicted_price2= self.Linear_estimator3.predict(x1)[0]
            
        
            predicted_price=0.2*predicted_price1+ 0.8*predicted_price2
            
            
        else:
            if flag==1:
                
                predicted_price1= self.random_estimator2.predict(test_row)[0] 
                x1 = test_row.full_sq
                x1 = x1.reshape(len(x1),1)
                predicted_price2= self.Linear_estimator2.predict(x1)[0]
                predicted_price=0.5*predicted_price1+ 0.5*predicted_price2
            
            else:
                if self.FLAG2==1:
                    testrow1=testrow1.drop(['sub_area', 'full_sq' ,'product_type'],axis=1)
                    testrow1=testrow1.iloc[:,2:20]
                    for fea in testrow1.columns:
                              testrow1[fea].fillna(testrow1[fea].median(),inplace=True)
                    for fea in testrow1.columns:
                              testrow1[fea].fillna(0,inplace=True)
                              
                    pclass=self.TwoClassModel.predict(testrow1)[0]
                
                else:
                       pclass=self.TwoClassModel.predict(test_row)[0]
                       
                if pclass==-1:
                       #predicted_price= 1.5
                       x1 = test_row.full_sq
                       x1 = x1.reshape(len(x1),1)
                       predicted_price= self.Linear_estimator4.predict(x1)[0]
                else:                        
                       predicted_price1= self.random_estimator3.predict(test_row)[0] 
                       x1 = test_row.full_sq
                       x1 = x1.reshape(len(x1),1)
                       predicted_price2= self.Linear_estimator3.predict(x1)[0]
                       predicted_price=.9*predicted_price1+ .1*predicted_price2
                       
        if self.specialFlag==1:
            #if (dum >70) & (dum <600) & (predicted_price > 10):
            #     predicted_price=predicted_price + 3
            if (predicted_price <3) :
                 predicted_price +=dum/50
        #predicted_price -= .1
        return predicted_price 

class Krasnoselskoe():
    #_data = pd.DataFrame
    def __init__(self, train_rows, twoclass):
        
        #### Flags
        self.specialFlag=0 ### this is for special adjustment in predict
        self.FLAG1=1  ### this is 1 for combined s2 s3
        self.FLAG2=1 ### this is 1 for global TwoClass classifier
        
        
        
        self.highcutoff = 150
        self.highcutoff1=150
        self.lowpricecutoff =5.2

        s1=train_rows[train_rows.full_sq > self.highcutoff];
        sa=train_rows[train_rows.full_sq < self.highcutoff];
        s2=sa[(sa.state.isnull()) & (sa.life_sq.isnull())]
        s3a = sa[~((sa.state.isnull()) & (sa.life_sq.isnull()))]
        s3=s3a[s3a.price_doc>self.lowpricecutoff]
        s4=s3a[s3a.price_doc <= self.lowpricecutoff]
        t1=s1.price_doc
        t2=s2.price_doc
        t3=s3.price_doc
        t4=s4.price_doc
        s1=s1.drop(['price_doc','sub_area','product_type'],axis=1)
        s1=s1.iloc[:,2:20]
        for fea in s1.columns:
                s1[fea].fillna(s1[fea].median(), inplace=True)
        for fea in s1.columns:
                s1[fea].fillna(0, inplace=True)
        s2=s2.drop(['price_doc','sub_area','product_type'],axis=1)
        s2=s2.iloc[:,2:20]
        for fea in s2.columns:
                s2[fea].fillna(s2[fea].median(), inplace=True)
        for fea in s2.columns:
                s2[fea].fillna(0, inplace=True)              
        s3=s3.drop(['price_doc','sub_area','product_type'],axis=1)
        s3=s3.iloc[:,2:20]
        for fea in s3.columns:
                s3[fea].fillna(s3[fea].median(), inplace=True)
        for fea in s3.columns:
                s3[fea].fillna(0, inplace=True)
        s4=s4.drop(['price_doc','sub_area','product_type'],axis=1)
        s4=s4.iloc[:,2:20]
        for fea in s4.columns:
                s4[fea].fillna(s4[fea].median(), inplace=True)
        for fea in s4.columns:
                s4[fea].fillna(0, inplace=True)
        
        sa=sa.drop(['price_doc','sub_area','product_type'],axis=1)
        sa=sa.iloc[:,2:20]
        for fea in sa.columns:
                sa[fea].fillna(sa[fea].median(), inplace=True)
        for fea in sa.columns:
                sa[fea].fillna(0, inplace=True)
                
                
        if self.FLAG1==1:
           X23 = pd.concat([s2,s3]) 
           y23 = np.concatenate([t2,t3],axis=0)
           random_estimator23=RandomForestRegressor(random_state=0, n_estimators=500)
           random_estimator23.fit(X23,y23)
           self.random_estimator2=random_estimator23
           self.random_estimator3=random_estimator23
           
           X23r=X23.full_sq
           X23r=X23r.reshape(len(X23r),1)
           y23r=y23.reshape(len(y23), 1)
           Linear_estimator23=LinearRegression()
           Linear_estimator23.fit(X23r, y23r)
           
           self.Linear_estimator2=Linear_estimator23
           self.Linear_estimator3=Linear_estimator23
           
        else:
            
            random_estimator2 = RandomForestRegressor(random_state=0, n_estimators=500)
            random_estimator2.fit(s2,t2)
            self.random_estimator2=random_estimator2
            
            Linear_estimator2 = LinearRegression()
            
            s2r=s2.full_sq
            s2r=s2r.reshape(len(s2r),1)
            t2r=t2
            t2r=t2r.reshape(len(t2r),1)
            Linear_estimator2.fit(s2r, t2r)
            self.Linear_estimator2=Linear_estimator2
            
            
            
            random_estimator3 = RandomForestRegressor(random_state=0, n_estimators=500)
            random_estimator3.fit(s3,t3)
            self.random_estimator3=random_estimator3
            
            Linear_estimator3 = LinearRegression()
            
            s3r=s3.full_sq
            s3r=s3r.reshape(len(s3r),1)
            t3r=t3
            t3r=t3r.reshape(len(t3r),1)
            Linear_estimator3.fit(s3r, t3r)
            self.Linear_estimator3=Linear_estimator3
            
            
            
           
        if self.FLAG2==1:
               self.TwoClassModel=twoclass
        else :
               
                    
                d3=np.ones(len(t3))
                d4=np.ones(len(t4))*-1
                          
                X = pd.concat([s3,s4])
                y = np.concatenate([d3,d4],axis=0)
                self.y=y
                self.X=X
                #print (X)
                #self.decision_clf = svm.SVC()
                TwoClassModel = RandomForestClassifier()
                TwoClassModel.fit(X, y)
                self.TwoClassModel=TwoClassModel
                
            
        Linear_estimator4 = LinearRegression()        
        s4r=s4.full_sq
        s4r=s4r.reshape(len(s4r),1)
        t4r=t4
        t4r=t4r.reshape(len(t4r),1)
        Linear_estimator4.fit(s4r, t4r)
        self.Linear_estimator4=Linear_estimator4             

        
    def predict_price(self,test_row):
        testrow1=test_row
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
        if dum > self.highcutoff1:
            #predicted_price=5
            predicted_price1= self.random_estimator3.predict(test_row)[0] 
            
            x1 = test_row.full_sq
            x1 = x1.reshape(len(x1),1)
            
            predicted_price2= self.Linear_estimator3.predict(x1)[0]
            
        
            predicted_price=0.2*predicted_price1+ 0.8*predicted_price2
            
            
        else:
            if flag==1:
                
                predicted_price1= self.random_estimator2.predict(test_row)[0] 
                x1 = test_row.full_sq
                x1 = x1.reshape(len(x1),1)
                predicted_price2= self.Linear_estimator2.predict(x1)[0]
                predicted_price=0.5*predicted_price1+ 0.5*predicted_price2
            
            else:
                if self.FLAG2==1:
                    testrow1=testrow1.drop(['sub_area', 'full_sq' ,'product_type'],axis=1)
                    testrow1=testrow1.iloc[:,2:20]
                    for fea in testrow1.columns:
                              testrow1[fea].fillna(testrow1[fea].median(),inplace=True)
                    for fea in testrow1.columns:
                              testrow1[fea].fillna(0,inplace=True)
                              
                    pclass=self.TwoClassModel.predict(testrow1)[0]
                
                else:
                       pclass=self.TwoClassModel.predict(test_row)[0]
                       
                if pclass==-1:
                       #predicted_price= 1.5
                       x1 = test_row.full_sq
                       x1 = x1.reshape(len(x1),1)
                       predicted_price= self.Linear_estimator4.predict(x1)[0]
                else:                        
                       predicted_price1= self.random_estimator3.predict(test_row)[0] 
                       x1 = test_row.full_sq
                       x1 = x1.reshape(len(x1),1)
                       predicted_price2= self.Linear_estimator3.predict(x1)[0]
                       predicted_price=.9*predicted_price1+ .1*predicted_price2
                       
        if self.specialFlag==1:
            #if (dum >70) & (dum <600) & (predicted_price > 10):
            #     predicted_price=predicted_price + 3
            if (predicted_price <3) :
                 predicted_price +=dum/50
        #predicted_price -= .1
        return predicted_price 

gc.collect()
###############################################################
class Zamoskvoreche():
    #_data = pd.DataFrame
    def __init__(self, train_rows, twoclass):
        
        #### Flags
        self.specialFlag=0 ### this is for special adjustment in predict
        self.FLAG1=1  ### this is 1 for combined s2 s3
        self.FLAG2=1 ### this is 1 for global TwoClass classifier
        
        
        
        self.highcutoff = 150
        self.highcutoff1=150
        self.lowpricecutoff =3

        s1=train_rows[train_rows.full_sq > self.highcutoff];
        sa=train_rows[train_rows.full_sq < self.highcutoff];
        s2=sa[(sa.state.isnull()) & (sa.life_sq.isnull())]
        s3a = sa[~((sa.state.isnull()) & (sa.life_sq.isnull()))]
        s3=s3a[s3a.price_doc>self.lowpricecutoff]
        s4=s3a[s3a.price_doc <= self.lowpricecutoff]
        t1=s1.price_doc
        t2=s2.price_doc
        t3=s3.price_doc
        t4=s4.price_doc
        s1=s1.drop(['price_doc','sub_area','product_type'],axis=1)
        s1=s1.iloc[:,2:20]
        for fea in s1.columns:
                s1[fea].fillna(s1[fea].median(), inplace=True)
        for fea in s1.columns:
                s1[fea].fillna(0, inplace=True)
        s2=s2.drop(['price_doc','sub_area','product_type'],axis=1)
        s2=s2.iloc[:,2:20]
        for fea in s2.columns:
                s2[fea].fillna(s2[fea].median(), inplace=True)
        for fea in s2.columns:
                s2[fea].fillna(0, inplace=True)              
        s3=s3.drop(['price_doc','sub_area','product_type'],axis=1)
        s3=s3.iloc[:,2:20]
        for fea in s3.columns:
                s3[fea].fillna(s3[fea].median(), inplace=True)
        for fea in s3.columns:
                s3[fea].fillna(0, inplace=True)
        s4=s4.drop(['price_doc','sub_area','product_type'],axis=1)
        s4=s4.iloc[:,2:20]
        for fea in s4.columns:
                s4[fea].fillna(s4[fea].median(), inplace=True)
        for fea in s4.columns:
                s4[fea].fillna(0, inplace=True)
        
        sa=sa.drop(['price_doc','sub_area','product_type'],axis=1)
        sa=sa.iloc[:,2:20]
        for fea in sa.columns:
                sa[fea].fillna(sa[fea].median(), inplace=True)
        for fea in sa.columns:
                sa[fea].fillna(0, inplace=True)
                
                
        if self.FLAG1==1:
           X23 = pd.concat([s2,s3]) 
           y23 = np.concatenate([t2,t3],axis=0)
           random_estimator23=RandomForestRegressor(random_state=0, n_estimators=500)
           random_estimator23.fit(X23,y23)
           self.random_estimator2=random_estimator23
           self.random_estimator3=random_estimator23
           
           X23r=X23.full_sq
           X23r=X23r.reshape(len(X23r),1)
           y23r=y23.reshape(len(y23), 1)
           Linear_estimator23=LinearRegression()
           Linear_estimator23.fit(X23r, y23r)
           
           self.Linear_estimator2=Linear_estimator23
           self.Linear_estimator3=Linear_estimator23
           
        else:
            
            random_estimator2 = RandomForestRegressor(random_state=0, n_estimators=500)
            random_estimator2.fit(s2,t2)
            self.random_estimator2=random_estimator2
            
            Linear_estimator2 = LinearRegression()
            
            s2r=s2.full_sq
            s2r=s2r.reshape(len(s2r),1)
            t2r=t2
            t2r=t2r.reshape(len(t2r),1)
            Linear_estimator2.fit(s2r, t2r)
            self.Linear_estimator2=Linear_estimator2
            
            
            
            random_estimator3 = RandomForestRegressor(random_state=0, n_estimators=500)
            random_estimator3.fit(s3,t3)
            self.random_estimator3=random_estimator3
            
            Linear_estimator3 = LinearRegression()
            
            s3r=s3.full_sq
            s3r=s3r.reshape(len(s3r),1)
            t3r=t3
            t3r=t3r.reshape(len(t3r),1)
            Linear_estimator3.fit(s3r, t3r)
            self.Linear_estimator3=Linear_estimator3
            
            
            
           
        if self.FLAG2==1:
               self.TwoClassModel=twoclass
        else :
               
                    
                d3=np.ones(len(t3))
                d4=np.ones(len(t4))*-1
                          
                X = pd.concat([s3,s4])
                y = np.concatenate([d3,d4],axis=0)
                self.y=y
                self.X=X
                #print (X)
                #self.decision_clf = svm.SVC()
                TwoClassModel = RandomForestClassifier()
                TwoClassModel.fit(X, y)
                self.TwoClassModel=TwoClassModel
                
            
        Linear_estimator4 = LinearRegression()        
        s4r=s4.full_sq
        s4r=s4r.reshape(len(s4r),1)
        t4r=t4
        t4r=t4r.reshape(len(t4r),1)
        Linear_estimator4.fit(s4r, t4r)
        self.Linear_estimator4=Linear_estimator4             

        
    def predict_price(self,test_row):
        testrow1=test_row
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
        if dum > self.highcutoff1:
            #predicted_price=5
            predicted_price1= self.random_estimator3.predict(test_row)[0] 
            
            x1 = test_row.full_sq
            x1 = x1.reshape(len(x1),1)
            
            predicted_price2= self.Linear_estimator3.predict(x1)[0]
            
        
            predicted_price=0.2*predicted_price1+ 0.8*predicted_price2
            
            
        else:
            if flag==1:
                
                predicted_price1= self.random_estimator2.predict(test_row)[0] 
                x1 = test_row.full_sq
                x1 = x1.reshape(len(x1),1)
                predicted_price2= self.Linear_estimator2.predict(x1)[0]
                predicted_price=0.5*predicted_price1+ 0.5*predicted_price2
            
            else:
                if self.FLAG2==1:
                    testrow1=testrow1.drop(['sub_area', 'full_sq' ,'product_type'],axis=1)
                    testrow1=testrow1.iloc[:,2:20]
                    for fea in testrow1.columns:
                              testrow1[fea].fillna(testrow1[fea].median(),inplace=True)
                    for fea in testrow1.columns:
                              testrow1[fea].fillna(0,inplace=True)
                              
                    pclass=self.TwoClassModel.predict(testrow1)[0]
                
                else:
                       pclass=self.TwoClassModel.predict(test_row)[0]
                       
                if pclass==-1:
                       #predicted_price= 1.5
                       x1 = test_row.full_sq
                       x1 = x1.reshape(len(x1),1)
                       predicted_price= self.Linear_estimator4.predict(x1)[0]
                else:                        
                       predicted_price1= self.random_estimator3.predict(test_row)[0] 
                       x1 = test_row.full_sq
                       x1 = x1.reshape(len(x1),1)
                       predicted_price2= self.Linear_estimator3.predict(x1)[0]
                       predicted_price=.9*predicted_price1+ .1*predicted_price2
                       
        if self.specialFlag==1:
            #if (dum >70) & (dum <600) & (predicted_price > 10):
            #     predicted_price=predicted_price + 3
            if (predicted_price <3) :
                 predicted_price +=dum/50
        #predicted_price -= .1
        return predicted_price 
gc.collect()
###############################################################
class Dmitrovskoe():
    #_data = pd.DataFrame
    def __init__(self, train_rows, twoclass):
        
        #### Flags
        self.specialFlag=0 ### this is for special adjustment in predict
        self.FLAG1=1  ### this is 1 for combined s2 s3
        self.FLAG2=0 ### this is 1 for global TwoClass classifier
        
        
        
        self.highcutoff = 150
        self.highcutoff1=150
        self.lowpricecutoff =3.8

        s1=train_rows[train_rows.full_sq > self.highcutoff];
        sa=train_rows[train_rows.full_sq < self.highcutoff];
        s2=sa[(sa.state.isnull()) & (sa.life_sq.isnull())]
        s3a = sa[~((sa.state.isnull()) & (sa.life_sq.isnull()))]
        s3=s3a[s3a.price_doc>self.lowpricecutoff]
        s4=s3a[s3a.price_doc <= self.lowpricecutoff]
        t1=s1.price_doc
        t2=s2.price_doc
        t3=s3.price_doc
        t4=s4.price_doc
        s1=s1.drop(['price_doc','sub_area','product_type'],axis=1)
        s1=s1.iloc[:,2:20]
        for fea in s1.columns:
                s1[fea].fillna(s1[fea].median(), inplace=True)
        for fea in s1.columns:
                s1[fea].fillna(0, inplace=True)
        s2=s2.drop(['price_doc','sub_area','product_type'],axis=1)
        s2=s2.iloc[:,2:20]
        for fea in s2.columns:
                s2[fea].fillna(s2[fea].median(), inplace=True)
        for fea in s2.columns:
                s2[fea].fillna(0, inplace=True)              
        s3=s3.drop(['price_doc','sub_area','product_type'],axis=1)
        s3=s3.iloc[:,2:20]
        for fea in s3.columns:
                s3[fea].fillna(s3[fea].median(), inplace=True)
        for fea in s3.columns:
                s3[fea].fillna(0, inplace=True)
        s4=s4.drop(['price_doc','sub_area','product_type'],axis=1)
        s4=s4.iloc[:,2:20]
        for fea in s4.columns:
                s4[fea].fillna(s4[fea].median(), inplace=True)
        for fea in s4.columns:
                s4[fea].fillna(0, inplace=True)
        
        sa=sa.drop(['price_doc','sub_area','product_type'],axis=1)
        sa=sa.iloc[:,2:20]
        for fea in sa.columns:
                sa[fea].fillna(sa[fea].median(), inplace=True)
        for fea in sa.columns:
                sa[fea].fillna(0, inplace=True)
                
                
        if self.FLAG1==1:
           X23 = pd.concat([s2,s3]) 
           y23 = np.concatenate([t2,t3],axis=0)
           random_estimator23=RandomForestRegressor(random_state=0, n_estimators=500)
           random_estimator23.fit(X23,y23)
           self.random_estimator2=random_estimator23
           self.random_estimator3=random_estimator23
           
           X23r=X23.full_sq
           X23r=X23r.reshape(len(X23r),1)
           y23r=y23.reshape(len(y23), 1)
           Linear_estimator23=LinearRegression()
           Linear_estimator23.fit(X23r, y23r)
           
           self.Linear_estimator2=Linear_estimator23
           self.Linear_estimator3=Linear_estimator23
           
        else:
            
            random_estimator2 = RandomForestRegressor(random_state=0, n_estimators=500)
            random_estimator2.fit(s2,t2)
            self.random_estimator2=random_estimator2
            
            Linear_estimator2 = LinearRegression()
            
            s2r=s2.full_sq
            s2r=s2r.reshape(len(s2r),1)
            t2r=t2
            t2r=t2r.reshape(len(t2r),1)
            Linear_estimator2.fit(s2r, t2r)
            self.Linear_estimator2=Linear_estimator2
            
            
            
            random_estimator3 = RandomForestRegressor(random_state=0, n_estimators=500)
            random_estimator3.fit(s3,t3)
            self.random_estimator3=random_estimator3
            
            Linear_estimator3 = LinearRegression()
            
            s3r=s3.full_sq
            s3r=s3r.reshape(len(s3r),1)
            t3r=t3
            t3r=t3r.reshape(len(t3r),1)
            Linear_estimator3.fit(s3r, t3r)
            self.Linear_estimator3=Linear_estimator3
            
            
            
           
        if self.FLAG2==1:
               self.TwoClassModel=twoclass
        else :
               
                    
                d3=np.ones(len(t3))
                d4=np.ones(len(t4))*-1
                          
                X = pd.concat([s3,s4])
                y = np.concatenate([d3,d4],axis=0)
                self.y=y
                self.X=X
                #print (X)
                #self.decision_clf = svm.SVC()
                TwoClassModel = RandomForestClassifier()
                TwoClassModel.fit(X, y)
                self.TwoClassModel=TwoClassModel
                
            
        Linear_estimator4 = LinearRegression()        
        s4r=s4.full_sq
        s4r=s4r.reshape(len(s4r),1)
        t4r=t4
        t4r=t4r.reshape(len(t4r),1)
        Linear_estimator4.fit(s4r, t4r)
        self.Linear_estimator4=Linear_estimator4             

        
    def predict_price(self,test_row):
        testrow1=test_row
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
        if dum > self.highcutoff1:
            #predicted_price=5
            predicted_price1= self.random_estimator3.predict(test_row)[0] 
            
            x1 = test_row.full_sq
            x1 = x1.reshape(len(x1),1)
            
            predicted_price2= self.Linear_estimator3.predict(x1)[0]
            
        
            predicted_price=0.2*predicted_price1+ 0.8*predicted_price2
            
            
        else:
            if flag==1:
                
                predicted_price1= self.random_estimator2.predict(test_row)[0] 
                x1 = test_row.full_sq
                x1 = x1.reshape(len(x1),1)
                predicted_price2= self.Linear_estimator2.predict(x1)[0]
                predicted_price=0.5*predicted_price1+ 0.5*predicted_price2
            
            else:
                if self.FLAG2==1:
                    testrow1=testrow1.drop(['sub_area', 'full_sq' ,'product_type'],axis=1)
                    testrow1=testrow1.iloc[:,2:20]
                    for fea in testrow1.columns:
                              testrow1[fea].fillna(testrow1[fea].median(),inplace=True)
                    for fea in testrow1.columns:
                              testrow1[fea].fillna(0,inplace=True)
                              
                    pclass=self.TwoClassModel.predict(testrow1)[0]
                
                else:
                       pclass=self.TwoClassModel.predict(test_row)[0]
                       
                if pclass==-1:
                       #predicted_price= 1.5
                       x1 = test_row.full_sq
                       x1 = x1.reshape(len(x1),1)
                       predicted_price= self.Linear_estimator4.predict(x1)[0]
                else:                        
                       predicted_price1= self.random_estimator3.predict(test_row)[0] 
                       x1 = test_row.full_sq
                       x1 = x1.reshape(len(x1),1)
                       predicted_price2= self.Linear_estimator3.predict(x1)[0]
                       predicted_price=.9*predicted_price1+ .1*predicted_price2
                       
        if self.specialFlag==1:
            #if (dum >70) & (dum <600) & (predicted_price > 10):
            #     predicted_price=predicted_price + 3
            if (predicted_price <3) :
                 predicted_price +=dum/50
        #predicted_price -= .1
        return predicted_price 
gc.collect()
###############################################################
class Mozhajskoe():
    #_data = pd.DataFrame
    def __init__(self, train_rows, twoclass):
        
        #### Flags
        self.specialFlag=0 ### this is for special adjustment in predict
        self.FLAG1=1  ### this is 1 for combined s2 s3
        self.FLAG2=0 ### this is 1 for global TwoClass classifier
        
        
        
        self.highcutoff = 150
        self.highcutoff1=150
        self.lowpricecutoff =3.8

        s1=train_rows[train_rows.full_sq > self.highcutoff];
        sa=train_rows[train_rows.full_sq < self.highcutoff];
        s2=sa[(sa.state.isnull()) & (sa.life_sq.isnull())]
        s3a = sa[~((sa.state.isnull()) & (sa.life_sq.isnull()))]
        s3=s3a[s3a.price_doc>self.lowpricecutoff]
        s4=s3a[s3a.price_doc <= self.lowpricecutoff]
        t1=s1.price_doc
        t2=s2.price_doc
        t3=s3.price_doc
        t4=s4.price_doc
        s1=s1.drop(['price_doc','sub_area','product_type'],axis=1)
        s1=s1.iloc[:,2:20]
        for fea in s1.columns:
                s1[fea].fillna(s1[fea].median(), inplace=True)
        for fea in s1.columns:
                s1[fea].fillna(0, inplace=True)
        s2=s2.drop(['price_doc','sub_area','product_type'],axis=1)
        s2=s2.iloc[:,2:20]
        for fea in s2.columns:
                s2[fea].fillna(s2[fea].median(), inplace=True)
        for fea in s2.columns:
                s2[fea].fillna(0, inplace=True)              
        s3=s3.drop(['price_doc','sub_area','product_type'],axis=1)
        s3=s3.iloc[:,2:20]
        for fea in s3.columns:
                s3[fea].fillna(s3[fea].median(), inplace=True)
        for fea in s3.columns:
                s3[fea].fillna(0, inplace=True)
        s4=s4.drop(['price_doc','sub_area','product_type'],axis=1)
        s4=s4.iloc[:,2:20]
        for fea in s4.columns:
                s4[fea].fillna(s4[fea].median(), inplace=True)
        for fea in s4.columns:
                s4[fea].fillna(0, inplace=True)
        
        sa=sa.drop(['price_doc','sub_area','product_type'],axis=1)
        sa=sa.iloc[:,2:20]
        for fea in sa.columns:
                sa[fea].fillna(sa[fea].median(), inplace=True)
        for fea in sa.columns:
                sa[fea].fillna(0, inplace=True)
                
                
        if self.FLAG1==1:
           X23 = pd.concat([s2,s3]) 
           y23 = np.concatenate([t2,t3],axis=0)
           random_estimator23=RandomForestRegressor(random_state=0, n_estimators=500)
           random_estimator23.fit(X23,y23)
           self.random_estimator2=random_estimator23
           self.random_estimator3=random_estimator23
           
           X23r=X23.full_sq
           X23r=X23r.reshape(len(X23r),1)
           y23r=y23.reshape(len(y23), 1)
           Linear_estimator23=LinearRegression()
           Linear_estimator23.fit(X23r, y23r)
           
           self.Linear_estimator2=Linear_estimator23
           self.Linear_estimator3=Linear_estimator23
           
        else:
            
            random_estimator2 = RandomForestRegressor(random_state=0, n_estimators=500)
            random_estimator2.fit(s2,t2)
            self.random_estimator2=random_estimator2
            
            Linear_estimator2 = LinearRegression()
            
            s2r=s2.full_sq
            s2r=s2r.reshape(len(s2r),1)
            t2r=t2
            t2r=t2r.reshape(len(t2r),1)
            Linear_estimator2.fit(s2r, t2r)
            self.Linear_estimator2=Linear_estimator2
            
            
            
            random_estimator3 = RandomForestRegressor(random_state=0, n_estimators=500)
            random_estimator3.fit(s3,t3)
            self.random_estimator3=random_estimator3
            
            Linear_estimator3 = LinearRegression()
            
            s3r=s3.full_sq
            s3r=s3r.reshape(len(s3r),1)
            t3r=t3
            t3r=t3r.reshape(len(t3r),1)
            Linear_estimator3.fit(s3r, t3r)
            self.Linear_estimator3=Linear_estimator3
            
            
            
           
        if self.FLAG2==1:
               self.TwoClassModel=twoclass
        else :
               
                    
                d3=np.ones(len(t3))
                d4=np.ones(len(t4))*-1
                          
                X = pd.concat([s3,s4])
                y = np.concatenate([d3,d4],axis=0)
                self.y=y
                self.X=X
                #print (X)
                #self.decision_clf = svm.SVC()
                TwoClassModel = RandomForestClassifier()
                TwoClassModel.fit(X, y)
                self.TwoClassModel=TwoClassModel
                
            
        Linear_estimator4 = LinearRegression()        
        s4r=s4.full_sq
        s4r=s4r.reshape(len(s4r),1)
        t4r=t4
        t4r=t4r.reshape(len(t4r),1)
        Linear_estimator4.fit(s4r, t4r)
        self.Linear_estimator4=Linear_estimator4             

        
    def predict_price(self,test_row):
        testrow1=test_row
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
        if dum > self.highcutoff1:
            #predicted_price=5
            predicted_price1= self.random_estimator3.predict(test_row)[0] 
            
            x1 = test_row.full_sq
            x1 = x1.reshape(len(x1),1)
            
            predicted_price2= self.Linear_estimator3.predict(x1)[0]
            
        
            predicted_price=0.2*predicted_price1+ 0.8*predicted_price2
            
            
        else:
            if flag==1:
                
                predicted_price1= self.random_estimator2.predict(test_row)[0] 
                x1 = test_row.full_sq
                x1 = x1.reshape(len(x1),1)
                predicted_price2= self.Linear_estimator2.predict(x1)[0]
                predicted_price=0.5*predicted_price1+ 0.5*predicted_price2
            
            else:
                if self.FLAG2==1:
                    testrow1=testrow1.drop(['sub_area', 'full_sq' ,'product_type'],axis=1)
                    testrow1=testrow1.iloc[:,2:20]
                    for fea in testrow1.columns:
                              testrow1[fea].fillna(testrow1[fea].median(),inplace=True)
                    for fea in testrow1.columns:
                              testrow1[fea].fillna(0,inplace=True)
                              
                    pclass=self.TwoClassModel.predict(testrow1)[0]
                
                else:
                       pclass=self.TwoClassModel.predict(test_row)[0]
                       
                if pclass==-1:
                       #predicted_price= 1.5
                       x1 = test_row.full_sq
                       x1 = x1.reshape(len(x1),1)
                       predicted_price= self.Linear_estimator4.predict(x1)[0]
                else:                        
                       predicted_price1= self.random_estimator3.predict(test_row)[0] 
                       x1 = test_row.full_sq
                       x1 = x1.reshape(len(x1),1)
                       predicted_price2= self.Linear_estimator3.predict(x1)[0]
                       predicted_price=.9*predicted_price1+ .1*predicted_price2
                       
        if self.specialFlag==1:
            #if (dum >70) & (dum <600) & (predicted_price > 10):
            #     predicted_price=predicted_price + 3
            if (predicted_price <3) :
                 predicted_price +=dum/50
        #predicted_price -= .1
        return predicted_price 

gc.collect()
###############################################################
class Juzhnoe_Medvedkovo():
    #_data = pd.DataFrame
    def __init__(self, train_rows, twoclass):
        
        #### Flags
        self.specialFlag=0 ### this is for special adjustment in predict
        self.FLAG1=1  ### this is 1 for combined s2 s3
        self.FLAG2=0 ### this is 1 for global TwoClass classifier
        
        
        
        self.highcutoff = 150
        self.highcutoff1=150
        self.lowpricecutoff =4.8

        s1=train_rows[train_rows.full_sq > self.highcutoff];
        sa=train_rows[train_rows.full_sq < self.highcutoff];
        s2=sa[(sa.state.isnull()) & (sa.life_sq.isnull())]
        s3a = sa[~((sa.state.isnull()) & (sa.life_sq.isnull()))]
        s3=s3a[s3a.price_doc>self.lowpricecutoff]
        s4=s3a[s3a.price_doc <= self.lowpricecutoff]
        t1=s1.price_doc
        t2=s2.price_doc
        t3=s3.price_doc
        t4=s4.price_doc
        s1=s1.drop(['price_doc','sub_area','product_type'],axis=1)
        s1=s1.iloc[:,2:20]
        for fea in s1.columns:
                s1[fea].fillna(s1[fea].median(), inplace=True)
        for fea in s1.columns:
                s1[fea].fillna(0, inplace=True)
        s2=s2.drop(['price_doc','sub_area','product_type'],axis=1)
        s2=s2.iloc[:,2:20]
        for fea in s2.columns:
                s2[fea].fillna(s2[fea].median(), inplace=True)
        for fea in s2.columns:
                s2[fea].fillna(0, inplace=True)              
        s3=s3.drop(['price_doc','sub_area','product_type'],axis=1)
        s3=s3.iloc[:,2:20]
        for fea in s3.columns:
                s3[fea].fillna(s3[fea].median(), inplace=True)
        for fea in s3.columns:
                s3[fea].fillna(0, inplace=True)
        s4=s4.drop(['price_doc','sub_area','product_type'],axis=1)
        s4=s4.iloc[:,2:20]
        for fea in s4.columns:
                s4[fea].fillna(s4[fea].median(), inplace=True)
        for fea in s4.columns:
                s4[fea].fillna(0, inplace=True)
        
        sa=sa.drop(['price_doc','sub_area','product_type'],axis=1)
        sa=sa.iloc[:,2:20]
        for fea in sa.columns:
                sa[fea].fillna(sa[fea].median(), inplace=True)
        for fea in sa.columns:
                sa[fea].fillna(0, inplace=True)
                
                
        if self.FLAG1==1:
           X23 = pd.concat([s2,s3]) 
           y23 = np.concatenate([t2,t3],axis=0)
           random_estimator23=RandomForestRegressor(random_state=0, n_estimators=500)
           random_estimator23.fit(X23,y23)
           self.random_estimator2=random_estimator23
           self.random_estimator3=random_estimator23
           
           X23r=X23.full_sq
           X23r=X23r.reshape(len(X23r),1)
           y23r=y23.reshape(len(y23), 1)
           Linear_estimator23=LinearRegression()
           Linear_estimator23.fit(X23r, y23r)
           
           self.Linear_estimator2=Linear_estimator23
           self.Linear_estimator3=Linear_estimator23
           
        else:
            
            random_estimator2 = RandomForestRegressor(random_state=0, n_estimators=500)
            random_estimator2.fit(s2,t2)
            self.random_estimator2=random_estimator2
            
            Linear_estimator2 = LinearRegression()
            
            s2r=s2.full_sq
            s2r=s2r.reshape(len(s2r),1)
            t2r=t2
            t2r=t2r.reshape(len(t2r),1)
            Linear_estimator2.fit(s2r, t2r)
            self.Linear_estimator2=Linear_estimator2
            
            
            
            random_estimator3 = RandomForestRegressor(random_state=0, n_estimators=500)
            random_estimator3.fit(s3,t3)
            self.random_estimator3=random_estimator3
            
            Linear_estimator3 = LinearRegression()
            
            s3r=s3.full_sq
            s3r=s3r.reshape(len(s3r),1)
            t3r=t3
            t3r=t3r.reshape(len(t3r),1)
            Linear_estimator3.fit(s3r, t3r)
            self.Linear_estimator3=Linear_estimator3
            
            
            
           
        if self.FLAG2==1:
               self.TwoClassModel=twoclass
        else :
               
                    
                d3=np.ones(len(t3))
                d4=np.ones(len(t4))*-1
                          
                X = pd.concat([s3,s4])
                y = np.concatenate([d3,d4],axis=0)
                self.y=y
                self.X=X
                #print (X)
                #self.decision_clf = svm.SVC()
                TwoClassModel = RandomForestClassifier()
                TwoClassModel.fit(X, y)
                self.TwoClassModel=TwoClassModel
                
            
        Linear_estimator4 = LinearRegression()        
        s4r=s4.full_sq
        s4r=s4r.reshape(len(s4r),1)
        t4r=t4
        t4r=t4r.reshape(len(t4r),1)
        Linear_estimator4.fit(s4r, t4r)
        self.Linear_estimator4=Linear_estimator4             

        
    def predict_price(self,test_row):
        testrow1=test_row
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
        if dum > self.highcutoff1:
            #predicted_price=5
            predicted_price1= self.random_estimator3.predict(test_row)[0] 
            
            x1 = test_row.full_sq
            x1 = x1.reshape(len(x1),1)
            
            predicted_price2= self.Linear_estimator3.predict(x1)[0]
            
        
            predicted_price=0.2*predicted_price1+ 0.8*predicted_price2
            
            
        else:
            if flag==1:
                
                predicted_price1= self.random_estimator2.predict(test_row)[0] 
                x1 = test_row.full_sq
                x1 = x1.reshape(len(x1),1)
                predicted_price2= self.Linear_estimator2.predict(x1)[0]
                predicted_price=0.5*predicted_price1+ 0.5*predicted_price2
            
            else:
                if self.FLAG2==1:
                    testrow1=testrow1.drop(['sub_area', 'full_sq' ,'product_type'],axis=1)
                    testrow1=testrow1.iloc[:,2:20]
                    for fea in testrow1.columns:
                              testrow1[fea].fillna(testrow1[fea].median(),inplace=True)
                    for fea in testrow1.columns:
                              testrow1[fea].fillna(0,inplace=True)
                              
                    pclass=self.TwoClassModel.predict(testrow1)[0]
                
                else:
                       pclass=self.TwoClassModel.predict(test_row)[0]
                       
                if pclass==-1:
                       #predicted_price= 1.5
                       x1 = test_row.full_sq
                       x1 = x1.reshape(len(x1),1)
                       predicted_price= self.Linear_estimator4.predict(x1)[0]
                else:                        
                       predicted_price1= self.random_estimator3.predict(test_row)[0] 
                       x1 = test_row.full_sq
                       x1 = x1.reshape(len(x1),1)
                       predicted_price2= self.Linear_estimator3.predict(x1)[0]
                       predicted_price=.9*predicted_price1+ .1*predicted_price2
                       
        if self.specialFlag==1:
            #if (dum >70) & (dum <600) & (predicted_price > 10):
            #     predicted_price=predicted_price + 3
            if (predicted_price <3) :
                 predicted_price +=dum/50
        #predicted_price -= .1
        return predicted_price 


gc.collect()
###############################################################
class Sokolinaja_Gora():
    #_data = pd.DataFrame
    def __init__(self, train_rows, twoclass):
        
        #### Flags
        self.specialFlag=0 ### this is for special adjustment in predict
        self.FLAG1=1  ### this is 1 for combined s2 s3
        self.FLAG2=0 ### this is 1 for global TwoClass classifier
        
        
        
        self.highcutoff = 150
        self.highcutoff1=150
        self.lowpricecutoff =2.2

        s1=train_rows[train_rows.full_sq > self.highcutoff];
        sa=train_rows[train_rows.full_sq < self.highcutoff];
        s2=sa[(sa.state.isnull()) & (sa.life_sq.isnull())]
        s3a = sa[~((sa.state.isnull()) & (sa.life_sq.isnull()))]
        s3=s3a[s3a.price_doc>self.lowpricecutoff]
        s4=s3a[s3a.price_doc <= self.lowpricecutoff]
        t1=s1.price_doc
        t2=s2.price_doc
        t3=s3.price_doc
        t4=s4.price_doc
        s1=s1.drop(['price_doc','sub_area','product_type'],axis=1)
        s1=s1.iloc[:,2:20]
        for fea in s1.columns:
                s1[fea].fillna(s1[fea].median(), inplace=True)
        for fea in s1.columns:
                s1[fea].fillna(0, inplace=True)
        s2=s2.drop(['price_doc','sub_area','product_type'],axis=1)
        s2=s2.iloc[:,2:20]
        for fea in s2.columns:
                s2[fea].fillna(s2[fea].median(), inplace=True)
        for fea in s2.columns:
                s2[fea].fillna(0, inplace=True)              
        s3=s3.drop(['price_doc','sub_area','product_type'],axis=1)
        s3=s3.iloc[:,2:20]
        for fea in s3.columns:
                s3[fea].fillna(s3[fea].median(), inplace=True)
        for fea in s3.columns:
                s3[fea].fillna(0, inplace=True)
        s4=s4.drop(['price_doc','sub_area','product_type'],axis=1)
        s4=s4.iloc[:,2:20]
        for fea in s4.columns:
                s4[fea].fillna(s4[fea].median(), inplace=True)
        for fea in s4.columns:
                s4[fea].fillna(0, inplace=True)
        
        sa=sa.drop(['price_doc','sub_area','product_type'],axis=1)
        sa=sa.iloc[:,2:20]
        for fea in sa.columns:
                sa[fea].fillna(sa[fea].median(), inplace=True)
        for fea in sa.columns:
                sa[fea].fillna(0, inplace=True)
                
                
        if self.FLAG1==1:
           X23 = pd.concat([s2,s3]) 
           y23 = np.concatenate([t2,t3],axis=0)
           random_estimator23=RandomForestRegressor(random_state=0, n_estimators=500)
           random_estimator23.fit(X23,y23)
           self.random_estimator2=random_estimator23
           self.random_estimator3=random_estimator23
           
           X23r=X23.full_sq
           X23r=X23r.reshape(len(X23r),1)
           y23r=y23.reshape(len(y23), 1)
           Linear_estimator23=LinearRegression()
           Linear_estimator23.fit(X23r, y23r)
           
           self.Linear_estimator2=Linear_estimator23
           self.Linear_estimator3=Linear_estimator23
           
        else:
            
            random_estimator2 = RandomForestRegressor(random_state=0, n_estimators=500)
            random_estimator2.fit(s2,t2)
            self.random_estimator2=random_estimator2
            
            Linear_estimator2 = LinearRegression()
            
            s2r=s2.full_sq
            s2r=s2r.reshape(len(s2r),1)
            t2r=t2
            t2r=t2r.reshape(len(t2r),1)
            Linear_estimator2.fit(s2r, t2r)
            self.Linear_estimator2=Linear_estimator2
            
            
            
            random_estimator3 = RandomForestRegressor(random_state=0, n_estimators=500)
            random_estimator3.fit(s3,t3)
            self.random_estimator3=random_estimator3
            
            Linear_estimator3 = LinearRegression()
            
            s3r=s3.full_sq
            s3r=s3r.reshape(len(s3r),1)
            t3r=t3
            t3r=t3r.reshape(len(t3r),1)
            Linear_estimator3.fit(s3r, t3r)
            self.Linear_estimator3=Linear_estimator3
            
            
            
           
        if self.FLAG2==1:
               self.TwoClassModel=twoclass
        else :
               
                    
                d3=np.ones(len(t3))
                d4=np.ones(len(t4))*-1
                          
                X = pd.concat([s3,s4])
                y = np.concatenate([d3,d4],axis=0)
                self.y=y
                self.X=X
                #print (X)
                #self.decision_clf = svm.SVC()
                TwoClassModel = RandomForestClassifier()
                TwoClassModel.fit(X, y)
                self.TwoClassModel=TwoClassModel
                
            
        Linear_estimator4 = LinearRegression()        
        s4r=s4.full_sq
        s4r=s4r.reshape(len(s4r),1)
        t4r=t4
        t4r=t4r.reshape(len(t4r),1)
        Linear_estimator4.fit(s4r, t4r)
        self.Linear_estimator4=Linear_estimator4             

        
    def predict_price(self,test_row):
        testrow1=test_row
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
        if dum > self.highcutoff1:
            #predicted_price=5
            predicted_price1= self.random_estimator3.predict(test_row)[0] 
            
            x1 = test_row.full_sq
            x1 = x1.reshape(len(x1),1)
            
            predicted_price2= self.Linear_estimator3.predict(x1)[0]
            
        
            predicted_price=0.2*predicted_price1+ 0.8*predicted_price2
            
            
        else:
            if flag==1:
                
                predicted_price1= self.random_estimator2.predict(test_row)[0] 
                x1 = test_row.full_sq
                x1 = x1.reshape(len(x1),1)
                predicted_price2= self.Linear_estimator2.predict(x1)[0]
                predicted_price=0.5*predicted_price1+ 0.5*predicted_price2
            
            else:
                if self.FLAG2==1:
                    testrow1=testrow1.drop(['sub_area', 'full_sq' ,'product_type'],axis=1)
                    testrow1=testrow1.iloc[:,2:20]
                    for fea in testrow1.columns:
                              testrow1[fea].fillna(testrow1[fea].median(),inplace=True)
                    for fea in testrow1.columns:
                              testrow1[fea].fillna(0,inplace=True)
                              
                    pclass=self.TwoClassModel.predict(testrow1)[0]
                
                else:
                       pclass=self.TwoClassModel.predict(test_row)[0]
                       
                if pclass==-1:
                       #predicted_price= 1.5
                       x1 = test_row.full_sq
                       x1 = x1.reshape(len(x1),1)
                       predicted_price= self.Linear_estimator4.predict(x1)[0]
                else:                        
                       predicted_price1= self.random_estimator3.predict(test_row)[0] 
                       x1 = test_row.full_sq
                       x1 = x1.reshape(len(x1),1)
                       predicted_price2= self.Linear_estimator3.predict(x1)[0]
                       predicted_price=.9*predicted_price1+ .1*predicted_price2
                       
        if self.specialFlag==1:
            #if (dum >70) & (dum <600) & (predicted_price > 10):
            #     predicted_price=predicted_price + 3
            if (predicted_price <3) :
                 predicted_price +=dum/50
        #predicted_price -= .1
        return predicted_price 

gc.collect()
###############################################################
class Lianozovo():
    #_data = pd.DataFrame
    def __init__(self, train_rows, twoclass):
        
        #### Flags
        self.specialFlag=0 ### this is for special adjustment in predict
        self.FLAG1=1  ### this is 1 for combined s2 s3
        self.FLAG2=0 ### this is 1 for global TwoClass classifier
        
        
        
        self.highcutoff = 150
        self.highcutoff1=150
        self.lowpricecutoff =4.9

        s1=train_rows[train_rows.full_sq > self.highcutoff];
        sa=train_rows[train_rows.full_sq < self.highcutoff];
        s2=sa[(sa.state.isnull()) & (sa.life_sq.isnull())]
        s3a = sa[~((sa.state.isnull()) & (sa.life_sq.isnull()))]
        s3=s3a[s3a.price_doc>self.lowpricecutoff]
        s4=s3a[s3a.price_doc <= self.lowpricecutoff]
        t1=s1.price_doc
        t2=s2.price_doc
        t3=s3.price_doc
        t4=s4.price_doc
        s1=s1.drop(['price_doc','sub_area','product_type'],axis=1)
        s1=s1.iloc[:,2:20]
        for fea in s1.columns:
                s1[fea].fillna(s1[fea].median(), inplace=True)
        for fea in s1.columns:
                s1[fea].fillna(0, inplace=True)
        s2=s2.drop(['price_doc','sub_area','product_type'],axis=1)
        s2=s2.iloc[:,2:20]
        for fea in s2.columns:
                s2[fea].fillna(s2[fea].median(), inplace=True)
        for fea in s2.columns:
                s2[fea].fillna(0, inplace=True)              
        s3=s3.drop(['price_doc','sub_area','product_type'],axis=1)
        s3=s3.iloc[:,2:20]
        for fea in s3.columns:
                s3[fea].fillna(s3[fea].median(), inplace=True)
        for fea in s3.columns:
                s3[fea].fillna(0, inplace=True)
        s4=s4.drop(['price_doc','sub_area','product_type'],axis=1)
        s4=s4.iloc[:,2:20]
        for fea in s4.columns:
                s4[fea].fillna(s4[fea].median(), inplace=True)
        for fea in s4.columns:
                s4[fea].fillna(0, inplace=True)
        
        sa=sa.drop(['price_doc','sub_area','product_type'],axis=1)
        sa=sa.iloc[:,2:20]
        for fea in sa.columns:
                sa[fea].fillna(sa[fea].median(), inplace=True)
        for fea in sa.columns:
                sa[fea].fillna(0, inplace=True)
                
                
        if self.FLAG1==1:
           X23 = pd.concat([s2,s3]) 
           y23 = np.concatenate([t2,t3],axis=0)
           random_estimator23=RandomForestRegressor(random_state=0, n_estimators=500)
           random_estimator23.fit(X23,y23)
           self.random_estimator2=random_estimator23
           self.random_estimator3=random_estimator23
           
           X23r=X23.full_sq
           X23r=X23r.reshape(len(X23r),1)
           y23r=y23.reshape(len(y23), 1)
           Linear_estimator23=LinearRegression()
           Linear_estimator23.fit(X23r, y23r)
           
           self.Linear_estimator2=Linear_estimator23
           self.Linear_estimator3=Linear_estimator23
           
        else:
            
            random_estimator2 = RandomForestRegressor(random_state=0, n_estimators=500)
            random_estimator2.fit(s2,t2)
            self.random_estimator2=random_estimator2
            
            Linear_estimator2 = LinearRegression()
            
            s2r=s2.full_sq
            s2r=s2r.reshape(len(s2r),1)
            t2r=t2
            t2r=t2r.reshape(len(t2r),1)
            Linear_estimator2.fit(s2r, t2r)
            self.Linear_estimator2=Linear_estimator2
            
            
            
            random_estimator3 = RandomForestRegressor(random_state=0, n_estimators=500)
            random_estimator3.fit(s3,t3)
            self.random_estimator3=random_estimator3
            
            Linear_estimator3 = LinearRegression()
            
            s3r=s3.full_sq
            s3r=s3r.reshape(len(s3r),1)
            t3r=t3
            t3r=t3r.reshape(len(t3r),1)
            Linear_estimator3.fit(s3r, t3r)
            self.Linear_estimator3=Linear_estimator3
            
            
            
           
        if self.FLAG2==1:
               self.TwoClassModel=twoclass
        else :
               
                    
                d3=np.ones(len(t3))
                d4=np.ones(len(t4))*-1
                          
                X = pd.concat([s3,s4])
                y = np.concatenate([d3,d4],axis=0)
                self.y=y
                self.X=X
                #print (X)
                #self.decision_clf = svm.SVC()
                TwoClassModel = RandomForestClassifier()
                TwoClassModel.fit(X, y)
                self.TwoClassModel=TwoClassModel
                
            
        Linear_estimator4 = LinearRegression()        
        s4r=s4.full_sq
        s4r=s4r.reshape(len(s4r),1)
        t4r=t4
        t4r=t4r.reshape(len(t4r),1)
        Linear_estimator4.fit(s4r, t4r)
        self.Linear_estimator4=Linear_estimator4             

        
    def predict_price(self,test_row):
        testrow1=test_row
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
        if dum > self.highcutoff1:
            #predicted_price=5
            predicted_price1= self.random_estimator3.predict(test_row)[0] 
            
            x1 = test_row.full_sq
            x1 = x1.reshape(len(x1),1)
            
            predicted_price2= self.Linear_estimator3.predict(x1)[0]
            
        
            predicted_price=0.2*predicted_price1+ 0.8*predicted_price2
            
            
        else:
            if flag==1:
                
                predicted_price1= self.random_estimator2.predict(test_row)[0] 
                x1 = test_row.full_sq
                x1 = x1.reshape(len(x1),1)
                predicted_price2= self.Linear_estimator2.predict(x1)[0]
                predicted_price=0.5*predicted_price1+ 0.5*predicted_price2
            
            else:
                if self.FLAG2==1:
                    testrow1=testrow1.drop(['sub_area', 'full_sq' ,'product_type'],axis=1)
                    testrow1=testrow1.iloc[:,2:20]
                    for fea in testrow1.columns:
                              testrow1[fea].fillna(testrow1[fea].median(),inplace=True)
                    for fea in testrow1.columns:
                              testrow1[fea].fillna(0,inplace=True)
                              
                    pclass=self.TwoClassModel.predict(testrow1)[0]
                
                else:
                       pclass=self.TwoClassModel.predict(test_row)[0]
                       
                if pclass==-1:
                       #predicted_price= 1.5
                       x1 = test_row.full_sq
                       x1 = x1.reshape(len(x1),1)
                       predicted_price= self.Linear_estimator4.predict(x1)[0]
                else:                        
                       predicted_price1= self.random_estimator3.predict(test_row)[0] 
                       x1 = test_row.full_sq
                       x1 = x1.reshape(len(x1),1)
                       predicted_price2= self.Linear_estimator3.predict(x1)[0]
                       predicted_price=.9*predicted_price1+ .1*predicted_price2
                       
        if self.specialFlag==1:
            #if (dum >70) & (dum <600) & (predicted_price > 10):
            #     predicted_price=predicted_price + 3
            if (predicted_price <3) :
                 predicted_price +=dum/50
        #predicted_price -= .1
        return predicted_price 

gc.collect()
###############################################################
class Pokrovskoe_Streshnevo():
    #_data = pd.DataFrame
    def __init__(self, train_rows, twoclass):
        
        #### Flags
        self.specialFlag=0 ### this is for special adjustment in predict
        self.FLAG1=0  ### this is 1 for combined s2 s3
        self.FLAG2=1 ### this is 1 for global TwoClass classifier
        
        
        
        self.highcutoff = 150
        self.highcutoff1=150
        self.lowpricecutoff =4.2

        s1=train_rows[train_rows.full_sq > self.highcutoff];
        sa=train_rows[train_rows.full_sq < self.highcutoff];
        s2=sa[(sa.state.isnull()) & (sa.life_sq.isnull())]
        s3a = sa[~((sa.state.isnull()) & (sa.life_sq.isnull()))]
        s3=s3a[s3a.price_doc>self.lowpricecutoff]
        s4=s3a[s3a.price_doc <= self.lowpricecutoff]
        t1=s1.price_doc
        t2=s2.price_doc
        t3=s3.price_doc
        t4=s4.price_doc
        s1=s1.drop(['price_doc','sub_area','product_type'],axis=1)
        s1=s1.iloc[:,2:20]
        for fea in s1.columns:
                s1[fea].fillna(s1[fea].median(), inplace=True)
        for fea in s1.columns:
                s1[fea].fillna(0, inplace=True)
        s2=s2.drop(['price_doc','sub_area','product_type'],axis=1)
        s2=s2.iloc[:,2:20]
        for fea in s2.columns:
                s2[fea].fillna(s2[fea].median(), inplace=True)
        for fea in s2.columns:
                s2[fea].fillna(0, inplace=True)              
        s3=s3.drop(['price_doc','sub_area','product_type'],axis=1)
        s3=s3.iloc[:,2:20]
        for fea in s3.columns:
                s3[fea].fillna(s3[fea].median(), inplace=True)
        for fea in s3.columns:
                s3[fea].fillna(0, inplace=True)
        s4=s4.drop(['price_doc','sub_area','product_type'],axis=1)
        s4=s4.iloc[:,2:20]
        for fea in s4.columns:
                s4[fea].fillna(s4[fea].median(), inplace=True)
        for fea in s4.columns:
                s4[fea].fillna(0, inplace=True)
        
        sa=sa.drop(['price_doc','sub_area','product_type'],axis=1)
        sa=sa.iloc[:,2:20]
        for fea in sa.columns:
                sa[fea].fillna(sa[fea].median(), inplace=True)
        for fea in sa.columns:
                sa[fea].fillna(0, inplace=True)
                
                
        if self.FLAG1==1:
           X23 = pd.concat([s2,s3]) 
           y23 = np.concatenate([t2,t3],axis=0)
           random_estimator23=RandomForestRegressor(random_state=0, n_estimators=500)
           random_estimator23.fit(X23,y23)
           self.random_estimator2=random_estimator23
           self.random_estimator3=random_estimator23
           
           X23r=X23.full_sq
           X23r=X23r.reshape(len(X23r),1)
           y23r=y23.reshape(len(y23), 1)
           Linear_estimator23=LinearRegression()
           Linear_estimator23.fit(X23r, y23r)
           
           self.Linear_estimator2=Linear_estimator23
           self.Linear_estimator3=Linear_estimator23
           
        else:
            
            random_estimator2 = RandomForestRegressor(random_state=0, n_estimators=500)
            random_estimator2.fit(s2,t2)
            self.random_estimator2=random_estimator2
            
            Linear_estimator2 = LinearRegression()
            
            s2r=s2.full_sq
            s2r=s2r.reshape(len(s2r),1)
            t2r=t2
            t2r=t2r.reshape(len(t2r),1)
            Linear_estimator2.fit(s2r, t2r)
            self.Linear_estimator2=Linear_estimator2
            
            
            
            random_estimator3 = RandomForestRegressor(random_state=0, n_estimators=500)
            random_estimator3.fit(s3,t3)
            self.random_estimator3=random_estimator3
            
            Linear_estimator3 = LinearRegression()
            
            s3r=s3.full_sq
            s3r=s3r.reshape(len(s3r),1)
            t3r=t3
            t3r=t3r.reshape(len(t3r),1)
            Linear_estimator3.fit(s3r, t3r)
            self.Linear_estimator3=Linear_estimator3
            
            
            
           
        if self.FLAG2==1:
               self.TwoClassModel=twoclass
        else :
               
                    
                d3=np.ones(len(t3))
                d4=np.ones(len(t4))*-1
                          
                X = pd.concat([s3,s4])
                y = np.concatenate([d3,d4],axis=0)
                self.y=y
                self.X=X
                #print (X)
                #self.decision_clf = svm.SVC()
                TwoClassModel = RandomForestClassifier()
                TwoClassModel.fit(X, y)
                self.TwoClassModel=TwoClassModel
                
            
        Linear_estimator4 = LinearRegression()        
        s4r=s4.full_sq
        s4r=s4r.reshape(len(s4r),1)
        t4r=t4
        t4r=t4r.reshape(len(t4r),1)
        Linear_estimator4.fit(s4r, t4r)
        self.Linear_estimator4=Linear_estimator4             

        
    def predict_price(self,test_row):
        testrow1=test_row
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
        if dum > self.highcutoff1:
            #predicted_price=5
            predicted_price1= self.random_estimator3.predict(test_row)[0] 
            
            x1 = test_row.full_sq
            x1 = x1.reshape(len(x1),1)
            
            predicted_price2= self.Linear_estimator3.predict(x1)[0]
            
        
            predicted_price=0.2*predicted_price1+ 0.8*predicted_price2
            
            
        else:
            if flag==1:
                
                predicted_price1= self.random_estimator2.predict(test_row)[0] 
                x1 = test_row.full_sq
                x1 = x1.reshape(len(x1),1)
                predicted_price2= self.Linear_estimator2.predict(x1)[0]
                predicted_price=0.5*predicted_price1+ 0.5*predicted_price2
            
            else:
                if self.FLAG2==1:
                    testrow1=testrow1.drop(['sub_area', 'full_sq' ,'product_type'],axis=1)
                    testrow1=testrow1.iloc[:,2:20]
                    for fea in testrow1.columns:
                              testrow1[fea].fillna(testrow1[fea].median(),inplace=True)
                    for fea in testrow1.columns:
                              testrow1[fea].fillna(0,inplace=True)
                              
                    pclass=self.TwoClassModel.predict(testrow1)[0]
                
                else:
                       pclass=self.TwoClassModel.predict(test_row)[0]
                       
                if pclass==-1:
                       #predicted_price= 1.5
                       x1 = test_row.full_sq
                       x1 = x1.reshape(len(x1),1)
                       predicted_price= self.Linear_estimator4.predict(x1)[0]
                else:                        
                       predicted_price1= self.random_estimator3.predict(test_row)[0] 
                       x1 = test_row.full_sq
                       x1 = x1.reshape(len(x1),1)
                       predicted_price2= self.Linear_estimator3.predict(x1)[0]
                       predicted_price=.9*predicted_price1+ .1*predicted_price2
                       
        if self.specialFlag==1:
            #if (dum >70) & (dum <600) & (predicted_price > 10):
            #     predicted_price=predicted_price + 3
            if (predicted_price <3) :
                 predicted_price +=dum/50
        #predicted_price -= .1
        return predicted_price 

gc.collect()
###############################################################
class Zapadnoe_Degunino():
    #_data = pd.DataFrame
    def __init__(self, train_rows, twoclass):
        
        #### Flags
        self.specialFlag=0 ### this is for special adjustment in predict
        self.FLAG1=0  ### this is 1 for combined s2 s3
        self.FLAG2=1 ### this is 1 for global TwoClass classifier
        
        
        
        self.highcutoff = 150
        self.highcutoff1=150
        self.lowpricecutoff =3.2

        s1=train_rows[train_rows.full_sq > self.highcutoff];
        sa=train_rows[train_rows.full_sq < self.highcutoff];
        s2=sa[(sa.state.isnull()) & (sa.life_sq.isnull())]
        s3a = sa[~((sa.state.isnull()) & (sa.life_sq.isnull()))]
        s3=s3a[s3a.price_doc>self.lowpricecutoff]
        s4=s3a[s3a.price_doc <= self.lowpricecutoff]
        t1=s1.price_doc
        t2=s2.price_doc
        t3=s3.price_doc
        t4=s4.price_doc
        s1=s1.drop(['price_doc','sub_area','product_type'],axis=1)
        s1=s1.iloc[:,2:20]
        for fea in s1.columns:
                s1[fea].fillna(s1[fea].median(), inplace=True)
        for fea in s1.columns:
                s1[fea].fillna(0, inplace=True)
        s2=s2.drop(['price_doc','sub_area','product_type'],axis=1)
        s2=s2.iloc[:,2:20]
        for fea in s2.columns:
                s2[fea].fillna(s2[fea].median(), inplace=True)
        for fea in s2.columns:
                s2[fea].fillna(0, inplace=True)              
        s3=s3.drop(['price_doc','sub_area','product_type'],axis=1)
        s3=s3.iloc[:,2:20]
        for fea in s3.columns:
                s3[fea].fillna(s3[fea].median(), inplace=True)
        for fea in s3.columns:
                s3[fea].fillna(0, inplace=True)
        s4=s4.drop(['price_doc','sub_area','product_type'],axis=1)
        s4=s4.iloc[:,2:20]
        for fea in s4.columns:
                s4[fea].fillna(s4[fea].median(), inplace=True)
        for fea in s4.columns:
                s4[fea].fillna(0, inplace=True)
        
        sa=sa.drop(['price_doc','sub_area','product_type'],axis=1)
        sa=sa.iloc[:,2:20]
        for fea in sa.columns:
                sa[fea].fillna(sa[fea].median(), inplace=True)
        for fea in sa.columns:
                sa[fea].fillna(0, inplace=True)
                
                
        if self.FLAG1==1:
           X23 = pd.concat([s2,s3]) 
           y23 = np.concatenate([t2,t3],axis=0)
           random_estimator23=RandomForestRegressor(random_state=0, n_estimators=500)
           random_estimator23.fit(X23,y23)
           self.random_estimator2=random_estimator23
           self.random_estimator3=random_estimator23
           
           X23r=X23.full_sq
           X23r=X23r.reshape(len(X23r),1)
           y23r=y23.reshape(len(y23), 1)
           Linear_estimator23=LinearRegression()
           Linear_estimator23.fit(X23r, y23r)
           
           self.Linear_estimator2=Linear_estimator23
           self.Linear_estimator3=Linear_estimator23
           
        else:
            
            random_estimator2 = RandomForestRegressor(random_state=0, n_estimators=500)
            random_estimator2.fit(s2,t2)
            self.random_estimator2=random_estimator2
            
            Linear_estimator2 = LinearRegression()
            
            s2r=s2.full_sq
            s2r=s2r.reshape(len(s2r),1)
            t2r=t2
            t2r=t2r.reshape(len(t2r),1)
            Linear_estimator2.fit(s2r, t2r)
            self.Linear_estimator2=Linear_estimator2
            
            
            
            random_estimator3 = RandomForestRegressor(random_state=0, n_estimators=500)
            random_estimator3.fit(s3,t3)
            self.random_estimator3=random_estimator3
            
            Linear_estimator3 = LinearRegression()
            
            s3r=s3.full_sq
            s3r=s3r.reshape(len(s3r),1)
            t3r=t3
            t3r=t3r.reshape(len(t3r),1)
            Linear_estimator3.fit(s3r, t3r)
            self.Linear_estimator3=Linear_estimator3
            
            
            
           
        if self.FLAG2==1:
               self.TwoClassModel=twoclass
        else :
               
                    
                d3=np.ones(len(t3))
                d4=np.ones(len(t4))*-1
                          
                X = pd.concat([s3,s4])
                y = np.concatenate([d3,d4],axis=0)
                self.y=y
                self.X=X
                #print (X)
                #self.decision_clf = svm.SVC()
                TwoClassModel = RandomForestClassifier()
                TwoClassModel.fit(X, y)
                self.TwoClassModel=TwoClassModel
                
            
        Linear_estimator4 = LinearRegression()        
        s4r=s4.full_sq
        s4r=s4r.reshape(len(s4r),1)
        t4r=t4
        t4r=t4r.reshape(len(t4r),1)
        Linear_estimator4.fit(s4r, t4r)
        self.Linear_estimator4=Linear_estimator4             

        
    def predict_price(self,test_row):
        testrow1=test_row
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
        if dum > self.highcutoff1:
            #predicted_price=5
            predicted_price1= self.random_estimator3.predict(test_row)[0] 
            
            x1 = test_row.full_sq
            x1 = x1.reshape(len(x1),1)
            
            predicted_price2= self.Linear_estimator3.predict(x1)[0]
            
        
            predicted_price=0.2*predicted_price1+ 0.8*predicted_price2
            
            
        else:
            if flag==1:
                
                predicted_price1= self.random_estimator2.predict(test_row)[0] 
                x1 = test_row.full_sq
                x1 = x1.reshape(len(x1),1)
                predicted_price2= self.Linear_estimator2.predict(x1)[0]
                predicted_price=0.5*predicted_price1+ 0.5*predicted_price2
            
            else:
                if self.FLAG2==1:
                    testrow1=testrow1.drop(['sub_area', 'full_sq' ,'product_type'],axis=1)
                    testrow1=testrow1.iloc[:,2:20]
                    for fea in testrow1.columns:
                              testrow1[fea].fillna(testrow1[fea].median(),inplace=True)
                    for fea in testrow1.columns:
                              testrow1[fea].fillna(0,inplace=True)
                              
                    pclass=self.TwoClassModel.predict(testrow1)[0]
                
                else:
                       pclass=self.TwoClassModel.predict(test_row)[0]
                       
                if pclass==-1:
                       #predicted_price= 1.5
                       x1 = test_row.full_sq
                       x1 = x1.reshape(len(x1),1)
                       predicted_price= self.Linear_estimator4.predict(x1)[0]
                else:                        
                       predicted_price1= self.random_estimator3.predict(test_row)[0] 
                       x1 = test_row.full_sq
                       x1 = x1.reshape(len(x1),1)
                       predicted_price2= self.Linear_estimator3.predict(x1)[0]
                       predicted_price=.9*predicted_price1+ .1*predicted_price2
                       
        if self.specialFlag==1:
            #if (dum >70) & (dum <600) & (predicted_price > 10):
            #     predicted_price=predicted_price + 3
            if (predicted_price <3) :
                 predicted_price +=dum/50
        #predicted_price -= .1
        return predicted_price 

gc.collect()
###############################################################
class Ajeroport():
    #_data = pd.DataFrame
    def __init__(self, train_rows, twoclass):
        
        #### Flags
        self.specialFlag=0 ### this is for special adjustment in predict
        self.FLAG1=1  ### this is 1 for combined s2 s3
        self.FLAG2=1 ### this is 1 for global TwoClass classifier
        
        
        
        self.highcutoff = 150
        self.highcutoff1=150
        self.lowpricecutoff =4.9

        s1=train_rows[train_rows.full_sq > self.highcutoff];
        sa=train_rows[train_rows.full_sq < self.highcutoff];
        s2=sa[(sa.state.isnull()) & (sa.life_sq.isnull())]
        s3a = sa[~((sa.state.isnull()) & (sa.life_sq.isnull()))]
        s3=s3a[s3a.price_doc>self.lowpricecutoff]
        s4=s3a[s3a.price_doc <= self.lowpricecutoff]
        t1=s1.price_doc
        t2=s2.price_doc
        t3=s3.price_doc
        t4=s4.price_doc
        s1=s1.drop(['price_doc','sub_area','product_type'],axis=1)
        s1=s1.iloc[:,2:20]
        for fea in s1.columns:
                s1[fea].fillna(s1[fea].median(), inplace=True)
        for fea in s1.columns:
                s1[fea].fillna(0, inplace=True)
        s2=s2.drop(['price_doc','sub_area','product_type'],axis=1)
        s2=s2.iloc[:,2:20]
        for fea in s2.columns:
                s2[fea].fillna(s2[fea].median(), inplace=True)
        for fea in s2.columns:
                s2[fea].fillna(0, inplace=True)              
        s3=s3.drop(['price_doc','sub_area','product_type'],axis=1)
        s3=s3.iloc[:,2:20]
        for fea in s3.columns:
                s3[fea].fillna(s3[fea].median(), inplace=True)
        for fea in s3.columns:
                s3[fea].fillna(0, inplace=True)
        s4=s4.drop(['price_doc','sub_area','product_type'],axis=1)
        s4=s4.iloc[:,2:20]
        for fea in s4.columns:
                s4[fea].fillna(s4[fea].median(), inplace=True)
        for fea in s4.columns:
                s4[fea].fillna(0, inplace=True)
        
        sa=sa.drop(['price_doc','sub_area','product_type'],axis=1)
        sa=sa.iloc[:,2:20]
        for fea in sa.columns:
                sa[fea].fillna(sa[fea].median(), inplace=True)
        for fea in sa.columns:
                sa[fea].fillna(0, inplace=True)
                
                
        if self.FLAG1==1:
           X23 = pd.concat([s2,s3]) 
           y23 = np.concatenate([t2,t3],axis=0)
           random_estimator23=RandomForestRegressor(random_state=0, n_estimators=500)
           random_estimator23.fit(X23,y23)
           self.random_estimator2=random_estimator23
           self.random_estimator3=random_estimator23
           
           X23r=X23.full_sq
           X23r=X23r.reshape(len(X23r),1)
           y23r=y23.reshape(len(y23), 1)
           Linear_estimator23=LinearRegression()
           Linear_estimator23.fit(X23r, y23r)
           
           self.Linear_estimator2=Linear_estimator23
           self.Linear_estimator3=Linear_estimator23
           
        else:
            
            random_estimator2 = RandomForestRegressor(random_state=0, n_estimators=500)
            random_estimator2.fit(s2,t2)
            self.random_estimator2=random_estimator2
            
            Linear_estimator2 = LinearRegression()
            
            s2r=s2.full_sq
            s2r=s2r.reshape(len(s2r),1)
            t2r=t2
            t2r=t2r.reshape(len(t2r),1)
            Linear_estimator2.fit(s2r, t2r)
            self.Linear_estimator2=Linear_estimator2
            
            
            
            random_estimator3 = RandomForestRegressor(random_state=0, n_estimators=500)
            random_estimator3.fit(s3,t3)
            self.random_estimator3=random_estimator3
            
            Linear_estimator3 = LinearRegression()
            
            s3r=s3.full_sq
            s3r=s3r.reshape(len(s3r),1)
            t3r=t3
            t3r=t3r.reshape(len(t3r),1)
            Linear_estimator3.fit(s3r, t3r)
            self.Linear_estimator3=Linear_estimator3
            
            
            
           
        if self.FLAG2==1:
               self.TwoClassModel=twoclass
        else :
               
                    
                d3=np.ones(len(t3))
                d4=np.ones(len(t4))*-1
                          
                X = pd.concat([s3,s4])
                y = np.concatenate([d3,d4],axis=0)
                self.y=y
                self.X=X
                #print (X)
                #self.decision_clf = svm.SVC()
                TwoClassModel = RandomForestClassifier()
                TwoClassModel.fit(X, y)
                self.TwoClassModel=TwoClassModel
                
            
        Linear_estimator4 = LinearRegression()        
        s4r=s4.full_sq
        s4r=s4r.reshape(len(s4r),1)
        t4r=t4
        t4r=t4r.reshape(len(t4r),1)
        Linear_estimator4.fit(s4r, t4r)
        self.Linear_estimator4=Linear_estimator4             

        
    def predict_price(self,test_row):
        testrow1=test_row
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
        if dum > self.highcutoff1:
            #predicted_price=5
            predicted_price1= self.random_estimator3.predict(test_row)[0] 
            
            x1 = test_row.full_sq
            x1 = x1.reshape(len(x1),1)
            
            predicted_price2= self.Linear_estimator3.predict(x1)[0]
            
        
            predicted_price=0.2*predicted_price1+ 0.8*predicted_price2
            
            
        else:
            if flag==1:
                
                predicted_price1= self.random_estimator2.predict(test_row)[0] 
                x1 = test_row.full_sq
                x1 = x1.reshape(len(x1),1)
                predicted_price2= self.Linear_estimator2.predict(x1)[0]
                predicted_price=0.5*predicted_price1+ 0.5*predicted_price2
            
            else:
                if self.FLAG2==1:
                    testrow1=testrow1.drop(['sub_area', 'full_sq' ,'product_type'],axis=1)
                    testrow1=testrow1.iloc[:,2:20]
                    for fea in testrow1.columns:
                              testrow1[fea].fillna(testrow1[fea].median(),inplace=True)
                    for fea in testrow1.columns:
                              testrow1[fea].fillna(0,inplace=True)
                              
                    pclass=self.TwoClassModel.predict(testrow1)[0]
                
                else:
                       pclass=self.TwoClassModel.predict(test_row)[0]
                       
                if pclass==-1:
                       #predicted_price= 1.5
                       x1 = test_row.full_sq
                       x1 = x1.reshape(len(x1),1)
                       predicted_price= self.Linear_estimator4.predict(x1)[0]
                else:                        
                       predicted_price1= self.random_estimator3.predict(test_row)[0] 
                       x1 = test_row.full_sq
                       x1 = x1.reshape(len(x1),1)
                       predicted_price2= self.Linear_estimator3.predict(x1)[0]
                       predicted_price=.9*predicted_price1+ .1*predicted_price2
                       
        if self.specialFlag==1:
            #if (dum >70) & (dum <600) & (predicted_price > 10):
            #     predicted_price=predicted_price + 3
            if (predicted_price <3) :
                 predicted_price +=dum/50
        #predicted_price -= .1
        return predicted_price 

gc.collect()
###############################################################
class Novogireevo():
    #_data = pd.DataFrame
    def __init__(self, train_rows, twoclass):
        
        #### Flags
        self.specialFlag=0 ### this is for special adjustment in predict
        self.FLAG1=1  ### this is 1 for combined s2 s3
        self.FLAG2=1 ### this is 1 for global TwoClass classifier
        
        
        
        self.highcutoff = 150
        self.highcutoff1=150
        self.lowpricecutoff =4.2

        s1=train_rows[train_rows.full_sq > self.highcutoff];
        sa=train_rows[train_rows.full_sq < self.highcutoff];
        s2=sa[(sa.state.isnull()) & (sa.life_sq.isnull())]
        s3a = sa[~((sa.state.isnull()) & (sa.life_sq.isnull()))]
        s3=s3a[s3a.price_doc>self.lowpricecutoff]
        s4=s3a[s3a.price_doc <= self.lowpricecutoff]
        t1=s1.price_doc
        t2=s2.price_doc
        t3=s3.price_doc
        t4=s4.price_doc
        s1=s1.drop(['price_doc','sub_area','product_type'],axis=1)
        s1=s1.iloc[:,2:20]
        for fea in s1.columns:
                s1[fea].fillna(s1[fea].median(), inplace=True)
        for fea in s1.columns:
                s1[fea].fillna(0, inplace=True)
        s2=s2.drop(['price_doc','sub_area','product_type'],axis=1)
        s2=s2.iloc[:,2:20]
        for fea in s2.columns:
                s2[fea].fillna(s2[fea].median(), inplace=True)
        for fea in s2.columns:
                s2[fea].fillna(0, inplace=True)              
        s3=s3.drop(['price_doc','sub_area','product_type'],axis=1)
        s3=s3.iloc[:,2:20]
        for fea in s3.columns:
                s3[fea].fillna(s3[fea].median(), inplace=True)
        for fea in s3.columns:
                s3[fea].fillna(0, inplace=True)
        s4=s4.drop(['price_doc','sub_area','product_type'],axis=1)
        s4=s4.iloc[:,2:20]
        for fea in s4.columns:
                s4[fea].fillna(s4[fea].median(), inplace=True)
        for fea in s4.columns:
                s4[fea].fillna(0, inplace=True)
        
        sa=sa.drop(['price_doc','sub_area','product_type'],axis=1)
        sa=sa.iloc[:,2:20]
        for fea in sa.columns:
                sa[fea].fillna(sa[fea].median(), inplace=True)
        for fea in sa.columns:
                sa[fea].fillna(0, inplace=True)
                
                
        if self.FLAG1==1:
           X23 = pd.concat([s2,s3]) 
           y23 = np.concatenate([t2,t3],axis=0)
           random_estimator23=RandomForestRegressor(random_state=0, n_estimators=500)
           random_estimator23.fit(X23,y23)
           self.random_estimator2=random_estimator23
           self.random_estimator3=random_estimator23
           
           X23r=X23.full_sq
           X23r=X23r.reshape(len(X23r),1)
           y23r=y23.reshape(len(y23), 1)
           Linear_estimator23=LinearRegression()
           Linear_estimator23.fit(X23r, y23r)
           
           self.Linear_estimator2=Linear_estimator23
           self.Linear_estimator3=Linear_estimator23
           
        else:
            
            random_estimator2 = RandomForestRegressor(random_state=0, n_estimators=500)
            random_estimator2.fit(s2,t2)
            self.random_estimator2=random_estimator2
            
            Linear_estimator2 = LinearRegression()
            
            s2r=s2.full_sq
            s2r=s2r.reshape(len(s2r),1)
            t2r=t2
            t2r=t2r.reshape(len(t2r),1)
            Linear_estimator2.fit(s2r, t2r)
            self.Linear_estimator2=Linear_estimator2
            
            
            
            random_estimator3 = RandomForestRegressor(random_state=0, n_estimators=500)
            random_estimator3.fit(s3,t3)
            self.random_estimator3=random_estimator3
            
            Linear_estimator3 = LinearRegression()
            
            s3r=s3.full_sq
            s3r=s3r.reshape(len(s3r),1)
            t3r=t3
            t3r=t3r.reshape(len(t3r),1)
            Linear_estimator3.fit(s3r, t3r)
            self.Linear_estimator3=Linear_estimator3
            
            
            
           
        if self.FLAG2==1:
               self.TwoClassModel=twoclass
        else :
               
                    
                d3=np.ones(len(t3))
                d4=np.ones(len(t4))*-1
                          
                X = pd.concat([s3,s4])
                y = np.concatenate([d3,d4],axis=0)
                self.y=y
                self.X=X
                #print (X)
                #self.decision_clf = svm.SVC()
                TwoClassModel = RandomForestClassifier()
                TwoClassModel.fit(X, y)
                self.TwoClassModel=TwoClassModel
                
            
        Linear_estimator4 = LinearRegression()        
        s4r=s4.full_sq
        s4r=s4r.reshape(len(s4r),1)
        t4r=t4
        t4r=t4r.reshape(len(t4r),1)
        Linear_estimator4.fit(s4r, t4r)
        self.Linear_estimator4=Linear_estimator4             

        
    def predict_price(self,test_row):
        testrow1=test_row
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
        if dum > self.highcutoff1:
            #predicted_price=5
            predicted_price1= self.random_estimator3.predict(test_row)[0] 
            
            x1 = test_row.full_sq
            x1 = x1.reshape(len(x1),1)
            
            predicted_price2= self.Linear_estimator3.predict(x1)[0]
            
        
            predicted_price=0.2*predicted_price1+ 0.8*predicted_price2
            
            
        else:
            if flag==1:
                
                predicted_price1= self.random_estimator2.predict(test_row)[0] 
                x1 = test_row.full_sq
                x1 = x1.reshape(len(x1),1)
                predicted_price2= self.Linear_estimator2.predict(x1)[0]
                predicted_price=0.5*predicted_price1+ 0.5*predicted_price2
            
            else:
                if self.FLAG2==1:
                    testrow1=testrow1.drop(['sub_area', 'full_sq' ,'product_type'],axis=1)
                    testrow1=testrow1.iloc[:,2:20]
                    for fea in testrow1.columns:
                              testrow1[fea].fillna(testrow1[fea].median(),inplace=True)
                    for fea in testrow1.columns:
                              testrow1[fea].fillna(0,inplace=True)
                              
                    pclass=self.TwoClassModel.predict(testrow1)[0]
                
                else:
                       pclass=self.TwoClassModel.predict(test_row)[0]
                       
                if pclass==-1:
                       #predicted_price= 1.5
                       x1 = test_row.full_sq
                       x1 = x1.reshape(len(x1),1)
                       predicted_price= self.Linear_estimator4.predict(x1)[0]
                else:                        
                       predicted_price1= self.random_estimator3.predict(test_row)[0] 
                       x1 = test_row.full_sq
                       x1 = x1.reshape(len(x1),1)
                       predicted_price2= self.Linear_estimator3.predict(x1)[0]
                       predicted_price=.9*predicted_price1+ .1*predicted_price2
                       
        if self.specialFlag==1:
            #if (dum >70) & (dum <600) & (predicted_price > 10):
            #     predicted_price=predicted_price + 3
            if (predicted_price <3) :
                 predicted_price +=dum/50
        #predicted_price -= .1
        return predicted_price 

gc.collect()
###############################################################
class Goljanovo():
    #_data = pd.DataFrame
    def __init__(self, train_rows, twoclass):
        
        #### Flags
        self.specialFlag=0 ### this is for special adjustment in predict
        self.FLAG1=1  ### this is 1 for combined s2 s3
        self.FLAG2=1 ### this is 1 for global TwoClass classifier
        
        
        
        self.highcutoff = 150
        self.highcutoff1=150
        self.lowpricecutoff =3.5

        s1=train_rows[train_rows.full_sq > self.highcutoff];
        sa=train_rows[train_rows.full_sq < self.highcutoff];
        s2=sa[(sa.state.isnull()) & (sa.life_sq.isnull())]
        s3a = sa[~((sa.state.isnull()) & (sa.life_sq.isnull()))]
        s3=s3a[s3a.price_doc>self.lowpricecutoff]
        s4=s3a[s3a.price_doc <= self.lowpricecutoff]
        t1=s1.price_doc
        t2=s2.price_doc
        t3=s3.price_doc
        t4=s4.price_doc
        s1=s1.drop(['price_doc','sub_area','product_type'],axis=1)
        s1=s1.iloc[:,2:20]
        for fea in s1.columns:
                s1[fea].fillna(s1[fea].median(), inplace=True)
        for fea in s1.columns:
                s1[fea].fillna(0, inplace=True)
        s2=s2.drop(['price_doc','sub_area','product_type'],axis=1)
        s2=s2.iloc[:,2:20]
        for fea in s2.columns:
                s2[fea].fillna(s2[fea].median(), inplace=True)
        for fea in s2.columns:
                s2[fea].fillna(0, inplace=True)              
        s3=s3.drop(['price_doc','sub_area','product_type'],axis=1)
        s3=s3.iloc[:,2:20]
        for fea in s3.columns:
                s3[fea].fillna(s3[fea].median(), inplace=True)
        for fea in s3.columns:
                s3[fea].fillna(0, inplace=True)
        s4=s4.drop(['price_doc','sub_area','product_type'],axis=1)
        s4=s4.iloc[:,2:20]
        for fea in s4.columns:
                s4[fea].fillna(s4[fea].median(), inplace=True)
        for fea in s4.columns:
                s4[fea].fillna(0, inplace=True)
        
        sa=sa.drop(['price_doc','sub_area','product_type'],axis=1)
        sa=sa.iloc[:,2:20]
        for fea in sa.columns:
                sa[fea].fillna(sa[fea].median(), inplace=True)
        for fea in sa.columns:
                sa[fea].fillna(0, inplace=True)
                
                
        if self.FLAG1==1:
           X23 = pd.concat([s2,s3]) 
           y23 = np.concatenate([t2,t3],axis=0)
           random_estimator23=RandomForestRegressor(random_state=0, n_estimators=500)
           random_estimator23.fit(X23,y23)
           self.random_estimator2=random_estimator23
           self.random_estimator3=random_estimator23
           
           X23r=X23.full_sq
           X23r=X23r.reshape(len(X23r),1)
           y23r=y23.reshape(len(y23), 1)
           Linear_estimator23=LinearRegression()
           Linear_estimator23.fit(X23r, y23r)
           
           self.Linear_estimator2=Linear_estimator23
           self.Linear_estimator3=Linear_estimator23
           
        else:
            
            random_estimator2 = RandomForestRegressor(random_state=0, n_estimators=500)
            random_estimator2.fit(s2,t2)
            self.random_estimator2=random_estimator2
            
            Linear_estimator2 = LinearRegression()
            
            s2r=s2.full_sq
            s2r=s2r.reshape(len(s2r),1)
            t2r=t2
            t2r=t2r.reshape(len(t2r),1)
            Linear_estimator2.fit(s2r, t2r)
            self.Linear_estimator2=Linear_estimator2
            
            
            
            random_estimator3 = RandomForestRegressor(random_state=0, n_estimators=500)
            random_estimator3.fit(s3,t3)
            self.random_estimator3=random_estimator3
            
            Linear_estimator3 = LinearRegression()
            
            s3r=s3.full_sq
            s3r=s3r.reshape(len(s3r),1)
            t3r=t3
            t3r=t3r.reshape(len(t3r),1)
            Linear_estimator3.fit(s3r, t3r)
            self.Linear_estimator3=Linear_estimator3
            
            
            
           
        if self.FLAG2==1:
               self.TwoClassModel=twoclass
        else :
               
                    
                d3=np.ones(len(t3))
                d4=np.ones(len(t4))*-1
                          
                X = pd.concat([s3,s4])
                y = np.concatenate([d3,d4],axis=0)
                self.y=y
                self.X=X
                #print (X)
                #self.decision_clf = svm.SVC()
                TwoClassModel = RandomForestClassifier()
                TwoClassModel.fit(X, y)
                self.TwoClassModel=TwoClassModel
                
            
        Linear_estimator4 = LinearRegression()        
        s4r=s4.full_sq
        s4r=s4r.reshape(len(s4r),1)
        t4r=t4
        t4r=t4r.reshape(len(t4r),1)
        Linear_estimator4.fit(s4r, t4r)
        self.Linear_estimator4=Linear_estimator4             

        
    def predict_price(self,test_row):
        testrow1=test_row
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
        if dum > self.highcutoff1:
            #predicted_price=5
            predicted_price1= self.random_estimator3.predict(test_row)[0] 
            
            x1 = test_row.full_sq
            x1 = x1.reshape(len(x1),1)
            
            predicted_price2= self.Linear_estimator3.predict(x1)[0]
            
        
            predicted_price=0.2*predicted_price1+ 0.8*predicted_price2
            
            
        else:
            if flag==1:
                
                predicted_price1= self.random_estimator2.predict(test_row)[0] 
                x1 = test_row.full_sq
                x1 = x1.reshape(len(x1),1)
                predicted_price2= self.Linear_estimator2.predict(x1)[0]
                predicted_price=0.5*predicted_price1+ 0.5*predicted_price2
            
            else:
                if self.FLAG2==1:
                    testrow1=testrow1.drop(['sub_area', 'full_sq' ,'product_type'],axis=1)
                    testrow1=testrow1.iloc[:,2:20]
                    for fea in testrow1.columns:
                              testrow1[fea].fillna(testrow1[fea].median(),inplace=True)
                    for fea in testrow1.columns:
                              testrow1[fea].fillna(0,inplace=True)
                              
                    pclass=self.TwoClassModel.predict(testrow1)[0]
                
                else:
                       pclass=self.TwoClassModel.predict(test_row)[0]
                       
                if pclass==-1:
                       #predicted_price= 1.5
                       x1 = test_row.full_sq
                       x1 = x1.reshape(len(x1),1)
                       predicted_price= self.Linear_estimator4.predict(x1)[0]
                else:                        
                       predicted_price1= self.random_estimator3.predict(test_row)[0] 
                       x1 = test_row.full_sq
                       x1 = x1.reshape(len(x1),1)
                       predicted_price2= self.Linear_estimator3.predict(x1)[0]
                       predicted_price=.9*predicted_price1+ .1*predicted_price2
                       
        if self.specialFlag==1:
            #if (dum >70) & (dum <600) & (predicted_price > 10):
            #     predicted_price=predicted_price + 3
            if (predicted_price <3) :
                 predicted_price +=dum/50
        #predicted_price -= .1
        return predicted_price 

gc.collect()
###############################################################
class Ostankinskoe():
    #_data = pd.DataFrame
    def __init__(self, train_rows, twoclass):
        
        #### Flags
        self.specialFlag=0 ### this is for special adjustment in predict
        self.FLAG1=1  ### this is 1 for combined s2 s3
        self.FLAG2=0 ### this is 1 for global TwoClass classifier
        
        
        
        self.highcutoff = 150
        self.highcutoff1=150
        self.lowpricecutoff =3

        s1=train_rows[train_rows.full_sq > self.highcutoff];
        sa=train_rows[train_rows.full_sq < self.highcutoff];
        s2=sa[(sa.state.isnull()) & (sa.life_sq.isnull())]
        s3a = sa[~((sa.state.isnull()) & (sa.life_sq.isnull()))]
        s3=s3a[s3a.price_doc>self.lowpricecutoff]
        s4=s3a[s3a.price_doc <= self.lowpricecutoff]
        t1=s1.price_doc
        t2=s2.price_doc
        t3=s3.price_doc
        t4=s4.price_doc
        s1=s1.drop(['price_doc','sub_area','product_type'],axis=1)
        s1=s1.iloc[:,2:20]
        for fea in s1.columns:
                s1[fea].fillna(s1[fea].median(), inplace=True)
        for fea in s1.columns:
                s1[fea].fillna(0, inplace=True)
        s2=s2.drop(['price_doc','sub_area','product_type'],axis=1)
        s2=s2.iloc[:,2:20]
        for fea in s2.columns:
                s2[fea].fillna(s2[fea].median(), inplace=True)
        for fea in s2.columns:
                s2[fea].fillna(0, inplace=True)              
        s3=s3.drop(['price_doc','sub_area','product_type'],axis=1)
        s3=s3.iloc[:,2:20]
        for fea in s3.columns:
                s3[fea].fillna(s3[fea].median(), inplace=True)
        for fea in s3.columns:
                s3[fea].fillna(0, inplace=True)
        s4=s4.drop(['price_doc','sub_area','product_type'],axis=1)
        s4=s4.iloc[:,2:20]
        for fea in s4.columns:
                s4[fea].fillna(s4[fea].median(), inplace=True)
        for fea in s4.columns:
                s4[fea].fillna(0, inplace=True)
        
        sa=sa.drop(['price_doc','sub_area','product_type'],axis=1)
        sa=sa.iloc[:,2:20]
        for fea in sa.columns:
                sa[fea].fillna(sa[fea].median(), inplace=True)
        for fea in sa.columns:
                sa[fea].fillna(0, inplace=True)
                
                
        if self.FLAG1==1:
           X23 = pd.concat([s2,s3]) 
           y23 = np.concatenate([t2,t3],axis=0)
           random_estimator23=RandomForestRegressor(random_state=0, n_estimators=500)
           random_estimator23.fit(X23,y23)
           self.random_estimator2=random_estimator23
           self.random_estimator3=random_estimator23
           
           X23r=X23.full_sq
           X23r=X23r.reshape(len(X23r),1)
           y23r=y23.reshape(len(y23), 1)
           Linear_estimator23=LinearRegression()
           Linear_estimator23.fit(X23r, y23r)
           
           self.Linear_estimator2=Linear_estimator23
           self.Linear_estimator3=Linear_estimator23
           
        else:
            
            random_estimator2 = RandomForestRegressor(random_state=0, n_estimators=500)
            random_estimator2.fit(s2,t2)
            self.random_estimator2=random_estimator2
            
            Linear_estimator2 = LinearRegression()
            
            s2r=s2.full_sq
            s2r=s2r.reshape(len(s2r),1)
            t2r=t2
            t2r=t2r.reshape(len(t2r),1)
            Linear_estimator2.fit(s2r, t2r)
            self.Linear_estimator2=Linear_estimator2
            
            
            
            random_estimator3 = RandomForestRegressor(random_state=0, n_estimators=500)
            random_estimator3.fit(s3,t3)
            self.random_estimator3=random_estimator3
            
            Linear_estimator3 = LinearRegression()
            
            s3r=s3.full_sq
            s3r=s3r.reshape(len(s3r),1)
            t3r=t3
            t3r=t3r.reshape(len(t3r),1)
            Linear_estimator3.fit(s3r, t3r)
            self.Linear_estimator3=Linear_estimator3
            
            
            
           
        if self.FLAG2==1:
               self.TwoClassModel=twoclass
        else :
               
                    
                d3=np.ones(len(t3))
                d4=np.ones(len(t4))*-1
                          
                X = pd.concat([s3,s4])
                y = np.concatenate([d3,d4],axis=0)
                self.y=y
                self.X=X
                #print (X)
                #self.decision_clf = svm.SVC()
                TwoClassModel = RandomForestClassifier()
                TwoClassModel.fit(X, y)
                self.TwoClassModel=TwoClassModel
                
            
        Linear_estimator4 = LinearRegression()        
        s4r=s4.full_sq
        s4r=s4r.reshape(len(s4r),1)
        t4r=t4
        t4r=t4r.reshape(len(t4r),1)
        Linear_estimator4.fit(s4r, t4r)
        self.Linear_estimator4=Linear_estimator4             

        
    def predict_price(self,test_row):
        testrow1=test_row
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
        if dum > self.highcutoff1:
            #predicted_price=5
            predicted_price1= self.random_estimator3.predict(test_row)[0] 
            
            x1 = test_row.full_sq
            x1 = x1.reshape(len(x1),1)
            
            predicted_price2= self.Linear_estimator3.predict(x1)[0]
            
        
            predicted_price=0.2*predicted_price1+ 0.8*predicted_price2
            
            
        else:
            if flag==1:
                
                predicted_price1= self.random_estimator2.predict(test_row)[0] 
                x1 = test_row.full_sq
                x1 = x1.reshape(len(x1),1)
                predicted_price2= self.Linear_estimator2.predict(x1)[0]
                predicted_price=0.5*predicted_price1+ 0.5*predicted_price2
            
            else:
                if self.FLAG2==1:
                    testrow1=testrow1.drop(['sub_area', 'full_sq' ,'product_type'],axis=1)
                    testrow1=testrow1.iloc[:,2:20]
                    for fea in testrow1.columns:
                              testrow1[fea].fillna(testrow1[fea].median(),inplace=True)
                    for fea in testrow1.columns:
                              testrow1[fea].fillna(0,inplace=True)
                              
                    pclass=self.TwoClassModel.predict(testrow1)[0]
                
                else:
                       pclass=self.TwoClassModel.predict(test_row)[0]
                       
                if pclass==-1:
                       #predicted_price= 1.5
                       x1 = test_row.full_sq
                       x1 = x1.reshape(len(x1),1)
                       predicted_price= self.Linear_estimator4.predict(x1)[0]
                else:                        
                       predicted_price1= self.random_estimator3.predict(test_row)[0] 
                       x1 = test_row.full_sq
                       x1 = x1.reshape(len(x1),1)
                       predicted_price2= self.Linear_estimator3.predict(x1)[0]
                       predicted_price=.9*predicted_price1+ .1*predicted_price2
                       
        if self.specialFlag==1:
            #if (dum >70) & (dum <600) & (predicted_price > 10):
            #     predicted_price=predicted_price + 3
            if (predicted_price <3) :
                 predicted_price +=dum/50
        #predicted_price -= .1
        return predicted_price 

gc.collect()
###############################################################
class Bogorodskoe():
    #_data = pd.DataFrame
    def __init__(self, train_rows, twoclass):
        
        #### Flags
        self.specialFlag=0 ### this is for special adjustment in predict
        self.FLAG1=1  ### this is 1 for combined s2 s3
        self.FLAG2=0 ### this is 1 for global TwoClass classifier
        
        
        
        self.highcutoff = 150
        self.highcutoff1=150
        self.lowpricecutoff =3

        s1=train_rows[train_rows.full_sq > self.highcutoff];
        sa=train_rows[train_rows.full_sq < self.highcutoff];
        s2=sa[(sa.state.isnull()) & (sa.life_sq.isnull())]
        s3a = sa[~((sa.state.isnull()) & (sa.life_sq.isnull()))]
        s3=s3a[s3a.price_doc>self.lowpricecutoff]
        s4=s3a[s3a.price_doc <= self.lowpricecutoff]
        t1=s1.price_doc
        t2=s2.price_doc
        t3=s3.price_doc
        t4=s4.price_doc
        s1=s1.drop(['price_doc','sub_area','product_type'],axis=1)
        s1=s1.iloc[:,2:20]
        for fea in s1.columns:
                s1[fea].fillna(s1[fea].median(), inplace=True)
        for fea in s1.columns:
                s1[fea].fillna(0, inplace=True)
        s2=s2.drop(['price_doc','sub_area','product_type'],axis=1)
        s2=s2.iloc[:,2:20]
        for fea in s2.columns:
                s2[fea].fillna(s2[fea].median(), inplace=True)
        for fea in s2.columns:
                s2[fea].fillna(0, inplace=True)              
        s3=s3.drop(['price_doc','sub_area','product_type'],axis=1)
        s3=s3.iloc[:,2:20]
        for fea in s3.columns:
                s3[fea].fillna(s3[fea].median(), inplace=True)
        for fea in s3.columns:
                s3[fea].fillna(0, inplace=True)
        s4=s4.drop(['price_doc','sub_area','product_type'],axis=1)
        s4=s4.iloc[:,2:20]
        for fea in s4.columns:
                s4[fea].fillna(s4[fea].median(), inplace=True)
        for fea in s4.columns:
                s4[fea].fillna(0, inplace=True)
        
        sa=sa.drop(['price_doc','sub_area','product_type'],axis=1)
        sa=sa.iloc[:,2:20]
        for fea in sa.columns:
                sa[fea].fillna(sa[fea].median(), inplace=True)
        for fea in sa.columns:
                sa[fea].fillna(0, inplace=True)
                
                
        if self.FLAG1==1:
           X23 = pd.concat([s2,s3]) 
           y23 = np.concatenate([t2,t3],axis=0)
           random_estimator23=RandomForestRegressor(random_state=0, n_estimators=500)
           random_estimator23.fit(X23,y23)
           self.random_estimator2=random_estimator23
           self.random_estimator3=random_estimator23
           
           X23r=X23.full_sq
           X23r=X23r.reshape(len(X23r),1)
           y23r=y23.reshape(len(y23), 1)
           Linear_estimator23=LinearRegression()
           Linear_estimator23.fit(X23r, y23r)
           
           self.Linear_estimator2=Linear_estimator23
           self.Linear_estimator3=Linear_estimator23
           
        else:
            
            random_estimator2 = RandomForestRegressor(random_state=0, n_estimators=500)
            random_estimator2.fit(s2,t2)
            self.random_estimator2=random_estimator2
            
            Linear_estimator2 = LinearRegression()
            
            s2r=s2.full_sq
            s2r=s2r.reshape(len(s2r),1)
            t2r=t2
            t2r=t2r.reshape(len(t2r),1)
            Linear_estimator2.fit(s2r, t2r)
            self.Linear_estimator2=Linear_estimator2
            
            
            
            random_estimator3 = RandomForestRegressor(random_state=0, n_estimators=500)
            random_estimator3.fit(s3,t3)
            self.random_estimator3=random_estimator3
            
            Linear_estimator3 = LinearRegression()
            
            s3r=s3.full_sq
            s3r=s3r.reshape(len(s3r),1)
            t3r=t3
            t3r=t3r.reshape(len(t3r),1)
            Linear_estimator3.fit(s3r, t3r)
            self.Linear_estimator3=Linear_estimator3
            
            
            
           
        if self.FLAG2==1:
               self.TwoClassModel=twoclass
        else :
               
                    
                d3=np.ones(len(t3))
                d4=np.ones(len(t4))*-1
                          
                X = pd.concat([s3,s4])
                y = np.concatenate([d3,d4],axis=0)
                self.y=y
                self.X=X
                #print (X)
                #self.decision_clf = svm.SVC()
                TwoClassModel = RandomForestClassifier()
                TwoClassModel.fit(X, y)
                self.TwoClassModel=TwoClassModel
                
            
        Linear_estimator4 = LinearRegression()        
        s4r=s4.full_sq
        s4r=s4r.reshape(len(s4r),1)
        t4r=t4
        t4r=t4r.reshape(len(t4r),1)
        Linear_estimator4.fit(s4r, t4r)
        self.Linear_estimator4=Linear_estimator4             

        
    def predict_price(self,test_row):
        testrow1=test_row
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
        if dum > self.highcutoff1:
            #predicted_price=5
            predicted_price1= self.random_estimator3.predict(test_row)[0] 
            
            x1 = test_row.full_sq
            x1 = x1.reshape(len(x1),1)
            
            predicted_price2= self.Linear_estimator3.predict(x1)[0]
            
        
            predicted_price=0.2*predicted_price1+ 0.8*predicted_price2
            
            
        else:
            if flag==1:
                
                predicted_price1= self.random_estimator2.predict(test_row)[0] 
                x1 = test_row.full_sq
                x1 = x1.reshape(len(x1),1)
                predicted_price2= self.Linear_estimator2.predict(x1)[0]
                predicted_price=0.5*predicted_price1+ 0.5*predicted_price2
            
            else:
                if self.FLAG2==1:
                    testrow1=testrow1.drop(['sub_area', 'full_sq' ,'product_type'],axis=1)
                    testrow1=testrow1.iloc[:,2:20]
                    for fea in testrow1.columns:
                              testrow1[fea].fillna(testrow1[fea].median(),inplace=True)
                    for fea in testrow1.columns:
                              testrow1[fea].fillna(0,inplace=True)
                              
                    pclass=self.TwoClassModel.predict(testrow1)[0]
                
                else:
                       pclass=self.TwoClassModel.predict(test_row)[0]
                       
                if pclass==-1:
                       #predicted_price= 1.5
                       x1 = test_row.full_sq
                       x1 = x1.reshape(len(x1),1)
                       predicted_price= self.Linear_estimator4.predict(x1)[0]
                else:                        
                       predicted_price1= self.random_estimator3.predict(test_row)[0] 
                       x1 = test_row.full_sq
                       x1 = x1.reshape(len(x1),1)
                       predicted_price2= self.Linear_estimator3.predict(x1)[0]
                       predicted_price=.9*predicted_price1+ .1*predicted_price2
                       
        if self.specialFlag==1:
            #if (dum >70) & (dum <600) & (predicted_price > 10):
            #     predicted_price=predicted_price + 3
            if (predicted_price <3) :
                 predicted_price +=dum/50
        #predicted_price -= .1
        return predicted_price 

gc.collect()
###############################################################
class Savelovskoe():
    #_data = pd.DataFrame
    def __init__(self, train_rows, twoclass):
        
        #### Flags
        self.specialFlag=0 ### this is for special adjustment in predict
        self.FLAG1=1  ### this is 1 for combined s2 s3
        self.FLAG2=0 ### this is 1 for global TwoClass classifier
        
        
        
        self.highcutoff = 150
        self.highcutoff1=150
        self.lowpricecutoff =2.5

        s1=train_rows[train_rows.full_sq > self.highcutoff];
        sa=train_rows[train_rows.full_sq < self.highcutoff];
        s2=sa[(sa.state.isnull()) & (sa.life_sq.isnull())]
        s3a = sa[~((sa.state.isnull()) & (sa.life_sq.isnull()))]
        s3=s3a[s3a.price_doc>self.lowpricecutoff]
        s4=s3a[s3a.price_doc <= self.lowpricecutoff]
        t1=s1.price_doc
        t2=s2.price_doc
        t3=s3.price_doc
        t4=s4.price_doc
        s1=s1.drop(['price_doc','sub_area','product_type'],axis=1)
        s1=s1.iloc[:,2:20]
        for fea in s1.columns:
                s1[fea].fillna(s1[fea].median(), inplace=True)
        for fea in s1.columns:
                s1[fea].fillna(0, inplace=True)
        s2=s2.drop(['price_doc','sub_area','product_type'],axis=1)
        s2=s2.iloc[:,2:20]
        for fea in s2.columns:
                s2[fea].fillna(s2[fea].median(), inplace=True)
        for fea in s2.columns:
                s2[fea].fillna(0, inplace=True)              
        s3=s3.drop(['price_doc','sub_area','product_type'],axis=1)
        s3=s3.iloc[:,2:20]
        for fea in s3.columns:
                s3[fea].fillna(s3[fea].median(), inplace=True)
        for fea in s3.columns:
                s3[fea].fillna(0, inplace=True)
        s4=s4.drop(['price_doc','sub_area','product_type'],axis=1)
        s4=s4.iloc[:,2:20]
        for fea in s4.columns:
                s4[fea].fillna(s4[fea].median(), inplace=True)
        for fea in s4.columns:
                s4[fea].fillna(0, inplace=True)
        
        sa=sa.drop(['price_doc','sub_area','product_type'],axis=1)
        sa=sa.iloc[:,2:20]
        for fea in sa.columns:
                sa[fea].fillna(sa[fea].median(), inplace=True)
        for fea in sa.columns:
                sa[fea].fillna(0, inplace=True)
                
                
        if self.FLAG1==1:
           X23 = pd.concat([s2,s3]) 
           y23 = np.concatenate([t2,t3],axis=0)
           random_estimator23=RandomForestRegressor(random_state=0, n_estimators=500)
           random_estimator23.fit(X23,y23)
           self.random_estimator2=random_estimator23
           self.random_estimator3=random_estimator23
           
           X23r=X23.full_sq
           X23r=X23r.reshape(len(X23r),1)
           y23r=y23.reshape(len(y23), 1)
           Linear_estimator23=LinearRegression()
           Linear_estimator23.fit(X23r, y23r)
           
           self.Linear_estimator2=Linear_estimator23
           self.Linear_estimator3=Linear_estimator23
           
        else:
            
            random_estimator2 = RandomForestRegressor(random_state=0, n_estimators=500)
            random_estimator2.fit(s2,t2)
            self.random_estimator2=random_estimator2
            
            Linear_estimator2 = LinearRegression()
            
            s2r=s2.full_sq
            s2r=s2r.reshape(len(s2r),1)
            t2r=t2
            t2r=t2r.reshape(len(t2r),1)
            Linear_estimator2.fit(s2r, t2r)
            self.Linear_estimator2=Linear_estimator2
            
            
            
            random_estimator3 = RandomForestRegressor(random_state=0, n_estimators=500)
            random_estimator3.fit(s3,t3)
            self.random_estimator3=random_estimator3
            
            Linear_estimator3 = LinearRegression()
            
            s3r=s3.full_sq
            s3r=s3r.reshape(len(s3r),1)
            t3r=t3
            t3r=t3r.reshape(len(t3r),1)
            Linear_estimator3.fit(s3r, t3r)
            self.Linear_estimator3=Linear_estimator3
            
            
            
           
        if self.FLAG2==1:
               self.TwoClassModel=twoclass
        else :
               
                    
                d3=np.ones(len(t3))
                d4=np.ones(len(t4))*-1
                          
                X = pd.concat([s3,s4])
                y = np.concatenate([d3,d4],axis=0)
                self.y=y
                self.X=X
                #print (X)
                #self.decision_clf = svm.SVC()
                TwoClassModel = RandomForestClassifier()
                TwoClassModel.fit(X, y)
                self.TwoClassModel=TwoClassModel
                
            
        Linear_estimator4 = LinearRegression()        
        s4r=s4.full_sq
        s4r=s4r.reshape(len(s4r),1)
        t4r=t4
        t4r=t4r.reshape(len(t4r),1)
        Linear_estimator4.fit(s4r, t4r)
        self.Linear_estimator4=Linear_estimator4             

        
    def predict_price(self,test_row):
        testrow1=test_row
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
        if dum > self.highcutoff1:
            #predicted_price=5
            predicted_price1= self.random_estimator3.predict(test_row)[0] 
            
            x1 = test_row.full_sq
            x1 = x1.reshape(len(x1),1)
            
            predicted_price2= self.Linear_estimator3.predict(x1)[0]
            
        
            predicted_price=0.2*predicted_price1+ 0.8*predicted_price2
            
            
        else:
            if flag==1:
                
                predicted_price1= self.random_estimator2.predict(test_row)[0] 
                x1 = test_row.full_sq
                x1 = x1.reshape(len(x1),1)
                predicted_price2= self.Linear_estimator2.predict(x1)[0]
                predicted_price=0.5*predicted_price1+ 0.5*predicted_price2
            
            else:
                if self.FLAG2==1:
                    testrow1=testrow1.drop(['sub_area', 'full_sq' ,'product_type'],axis=1)
                    testrow1=testrow1.iloc[:,2:20]
                    for fea in testrow1.columns:
                              testrow1[fea].fillna(testrow1[fea].median(),inplace=True)
                    for fea in testrow1.columns:
                              testrow1[fea].fillna(0,inplace=True)
                              
                    pclass=self.TwoClassModel.predict(testrow1)[0]
                
                else:
                       pclass=self.TwoClassModel.predict(test_row)[0]
                       
                if pclass==-1:
                       #predicted_price= 1.5
                       x1 = test_row.full_sq
                       x1 = x1.reshape(len(x1),1)
                       predicted_price= self.Linear_estimator4.predict(x1)[0]
                else:                        
                       predicted_price1= self.random_estimator3.predict(test_row)[0] 
                       x1 = test_row.full_sq
                       x1 = x1.reshape(len(x1),1)
                       predicted_price2= self.Linear_estimator3.predict(x1)[0]
                       predicted_price=.9*predicted_price1+ .1*predicted_price2
                       
        if self.specialFlag==1:
            #if (dum >70) & (dum <600) & (predicted_price > 10):
            #     predicted_price=predicted_price + 3
            if (predicted_price <3) :
                 predicted_price +=dum/50
        #predicted_price -= .1
        return predicted_price 

gc.collect()
###############################################################
class Presnenskoe():
    #_data = pd.DataFrame
    def __init__(self, train_rows, twoclass):
        
        #### Flags
        self.specialFlag=0 ### this is for special adjustment in predict
        self.FLAG1=1  ### this is 1 for combined s2 s3
        self.FLAG2=1 ### this is 1 for global TwoClass classifier
        
        
        
        self.highcutoff = 150
        self.highcutoff1=150
        self.lowpricecutoff =1

        s1=train_rows[train_rows.full_sq > self.highcutoff];
        sa=train_rows[train_rows.full_sq < self.highcutoff];
        s2=sa[(sa.state.isnull()) & (sa.life_sq.isnull())]
        s3a = sa[~((sa.state.isnull()) & (sa.life_sq.isnull()))]
        s3=s3a[s3a.price_doc>self.lowpricecutoff]
        s4=s3a[s3a.price_doc <= self.lowpricecutoff]
        t1=s1.price_doc
        t2=s2.price_doc
        t3=s3.price_doc
        t4=s4.price_doc
        s1=s1.drop(['price_doc','sub_area','product_type'],axis=1)
        s1=s1.iloc[:,2:20]
        for fea in s1.columns:
                s1[fea].fillna(s1[fea].median(), inplace=True)
        for fea in s1.columns:
                s1[fea].fillna(0, inplace=True)
        s2=s2.drop(['price_doc','sub_area','product_type'],axis=1)
        s2=s2.iloc[:,2:20]
        for fea in s2.columns:
                s2[fea].fillna(s2[fea].median(), inplace=True)
        for fea in s2.columns:
                s2[fea].fillna(0, inplace=True)              
        s3=s3.drop(['price_doc','sub_area','product_type'],axis=1)
        s3=s3.iloc[:,2:20]
        for fea in s3.columns:
                s3[fea].fillna(s3[fea].median(), inplace=True)
        for fea in s3.columns:
                s3[fea].fillna(0, inplace=True)
        s4=s4.drop(['price_doc','sub_area','product_type'],axis=1)
        s4=s4.iloc[:,2:20]
        for fea in s4.columns:
                s4[fea].fillna(s4[fea].median(), inplace=True)
        for fea in s4.columns:
                s4[fea].fillna(0, inplace=True)
        
        sa=sa.drop(['price_doc','sub_area','product_type'],axis=1)
        sa=sa.iloc[:,2:20]
        for fea in sa.columns:
                sa[fea].fillna(sa[fea].median(), inplace=True)
        for fea in sa.columns:
                sa[fea].fillna(0, inplace=True)
                
                
        if self.FLAG1==1:
           X23 = pd.concat([s2,s3]) 
           y23 = np.concatenate([t2,t3],axis=0)
           random_estimator23=RandomForestRegressor(random_state=0, n_estimators=500)
           random_estimator23.fit(X23,y23)
           self.random_estimator2=random_estimator23
           self.random_estimator3=random_estimator23
           
           X23r=X23.full_sq
           X23r=X23r.reshape(len(X23r),1)
           y23r=y23.reshape(len(y23), 1)
           Linear_estimator23=LinearRegression()
           Linear_estimator23.fit(X23r, y23r)
           
           self.Linear_estimator2=Linear_estimator23
           self.Linear_estimator3=Linear_estimator23
           
        else:
            
            random_estimator2 = RandomForestRegressor(random_state=0, n_estimators=500)
            random_estimator2.fit(s2,t2)
            self.random_estimator2=random_estimator2
            
            Linear_estimator2 = LinearRegression()
            
            s2r=s2.full_sq
            s2r=s2r.reshape(len(s2r),1)
            t2r=t2
            t2r=t2r.reshape(len(t2r),1)
            Linear_estimator2.fit(s2r, t2r)
            self.Linear_estimator2=Linear_estimator2
            
            
            
            random_estimator3 = RandomForestRegressor(random_state=0, n_estimators=500)
            random_estimator3.fit(s3,t3)
            self.random_estimator3=random_estimator3
            
            Linear_estimator3 = LinearRegression()
            
            s3r=s3.full_sq
            s3r=s3r.reshape(len(s3r),1)
            t3r=t3
            t3r=t3r.reshape(len(t3r),1)
            Linear_estimator3.fit(s3r, t3r)
            self.Linear_estimator3=Linear_estimator3
            
            
            
           
        if self.FLAG2==1:
               self.TwoClassModel=twoclass
        else :
               
                    
                d3=np.ones(len(t3))
                d4=np.ones(len(t4))*-1
                          
                X = pd.concat([s3,s4])
                y = np.concatenate([d3,d4],axis=0)
                self.y=y
                self.X=X
                #print (X)
                #self.decision_clf = svm.SVC()
                TwoClassModel = RandomForestClassifier()
                TwoClassModel.fit(X, y)
                self.TwoClassModel=TwoClassModel
                
            
        Linear_estimator4 = LinearRegression()        
        s4r=s4.full_sq
        s4r=s4r.reshape(len(s4r),1)
        t4r=t4
        t4r=t4r.reshape(len(t4r),1)
        Linear_estimator4.fit(s4r, t4r)
        self.Linear_estimator4=Linear_estimator4             

        
    def predict_price(self,test_row):
        testrow1=test_row
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
        if dum > self.highcutoff1:
            #predicted_price=5
            predicted_price1= self.random_estimator3.predict(test_row)[0] 
            
            x1 = test_row.full_sq
            x1 = x1.reshape(len(x1),1)
            
            predicted_price2= self.Linear_estimator3.predict(x1)[0]
            
        
            predicted_price=0.2*predicted_price1+ 0.8*predicted_price2
            
            
        else:
            if flag==1:
                
                predicted_price1= self.random_estimator2.predict(test_row)[0] 
                x1 = test_row.full_sq
                x1 = x1.reshape(len(x1),1)
                predicted_price2= self.Linear_estimator2.predict(x1)[0]
                predicted_price=0.5*predicted_price1+ 0.5*predicted_price2
            
            else:
                if self.FLAG2==1:
                    testrow1=testrow1.drop(['sub_area', 'full_sq' ,'product_type'],axis=1)
                    testrow1=testrow1.iloc[:,2:20]
                    for fea in testrow1.columns:
                              testrow1[fea].fillna(testrow1[fea].median(),inplace=True)
                    for fea in testrow1.columns:
                              testrow1[fea].fillna(0,inplace=True)
                              
                    pclass=self.TwoClassModel.predict(testrow1)[0]
                
                else:
                       pclass=self.TwoClassModel.predict(test_row)[0]
                       
                if pclass==-1:
                       #predicted_price= 1.5
                       x1 = test_row.full_sq
                       x1 = x1.reshape(len(x1),1)
                       predicted_price= self.Linear_estimator4.predict(x1)[0]
                else:                        
                       predicted_price1= self.random_estimator3.predict(test_row)[0] 
                       x1 = test_row.full_sq
                       x1 = x1.reshape(len(x1),1)
                       predicted_price2= self.Linear_estimator3.predict(x1)[0]
                       predicted_price=.9*predicted_price1+ .1*predicted_price2
                       
        if self.specialFlag==1:
            #if (dum >70) & (dum <600) & (predicted_price > 10):
            #     predicted_price=predicted_price + 3
            if (predicted_price <3) :
                 predicted_price +=dum/50
        #predicted_price -= .1
        return predicted_price 

gc.collect()
###############################################################
class Alekseevskoe():
    #_data = pd.DataFrame
    def __init__(self, train_rows, twoclass):
        
        #### Flags
        self.specialFlag=0 ### this is for special adjustment in predict
        self.FLAG1=1  ### this is 1 for combined s2 s3
        self.FLAG2=0 ### this is 1 for global TwoClass classifier
        
        
        
        self.highcutoff = 150
        self.highcutoff1=150
        self.lowpricecutoff =3.2

        s1=train_rows[train_rows.full_sq > self.highcutoff];
        sa=train_rows[train_rows.full_sq < self.highcutoff];
        s2=sa[(sa.state.isnull()) & (sa.life_sq.isnull())]
        s3a = sa[~((sa.state.isnull()) & (sa.life_sq.isnull()))]
        s3=s3a[s3a.price_doc>self.lowpricecutoff]
        s4=s3a[s3a.price_doc <= self.lowpricecutoff]
        t1=s1.price_doc
        t2=s2.price_doc
        t3=s3.price_doc
        t4=s4.price_doc
        s1=s1.drop(['price_doc','sub_area','product_type'],axis=1)
        s1=s1.iloc[:,2:20]
        for fea in s1.columns:
                s1[fea].fillna(s1[fea].median(), inplace=True)
        for fea in s1.columns:
                s1[fea].fillna(0, inplace=True)
        s2=s2.drop(['price_doc','sub_area','product_type'],axis=1)
        s2=s2.iloc[:,2:20]
        for fea in s2.columns:
                s2[fea].fillna(s2[fea].median(), inplace=True)
        for fea in s2.columns:
                s2[fea].fillna(0, inplace=True)              
        s3=s3.drop(['price_doc','sub_area','product_type'],axis=1)
        s3=s3.iloc[:,2:20]
        for fea in s3.columns:
                s3[fea].fillna(s3[fea].median(), inplace=True)
        for fea in s3.columns:
                s3[fea].fillna(0, inplace=True)
        s4=s4.drop(['price_doc','sub_area','product_type'],axis=1)
        s4=s4.iloc[:,2:20]
        for fea in s4.columns:
                s4[fea].fillna(s4[fea].median(), inplace=True)
        for fea in s4.columns:
                s4[fea].fillna(0, inplace=True)
        
        sa=sa.drop(['price_doc','sub_area','product_type'],axis=1)
        sa=sa.iloc[:,2:20]
        for fea in sa.columns:
                sa[fea].fillna(sa[fea].median(), inplace=True)
        for fea in sa.columns:
                sa[fea].fillna(0, inplace=True)
                
                
        if self.FLAG1==1:
           X23 = pd.concat([s2,s3]) 
           y23 = np.concatenate([t2,t3],axis=0)
           random_estimator23=RandomForestRegressor(random_state=0, n_estimators=500)
           random_estimator23.fit(X23,y23)
           self.random_estimator2=random_estimator23
           self.random_estimator3=random_estimator23
           
           X23r=X23.full_sq
           X23r=X23r.reshape(len(X23r),1)
           y23r=y23.reshape(len(y23), 1)
           Linear_estimator23=LinearRegression()
           Linear_estimator23.fit(X23r, y23r)
           
           self.Linear_estimator2=Linear_estimator23
           self.Linear_estimator3=Linear_estimator23
           
        else:
            
            random_estimator2 = RandomForestRegressor(random_state=0, n_estimators=500)
            random_estimator2.fit(s2,t2)
            self.random_estimator2=random_estimator2
            
            Linear_estimator2 = LinearRegression()
            
            s2r=s2.full_sq
            s2r=s2r.reshape(len(s2r),1)
            t2r=t2
            t2r=t2r.reshape(len(t2r),1)
            Linear_estimator2.fit(s2r, t2r)
            self.Linear_estimator2=Linear_estimator2
            
            
            
            random_estimator3 = RandomForestRegressor(random_state=0, n_estimators=500)
            random_estimator3.fit(s3,t3)
            self.random_estimator3=random_estimator3
            
            Linear_estimator3 = LinearRegression()
            
            s3r=s3.full_sq
            s3r=s3r.reshape(len(s3r),1)
            t3r=t3
            t3r=t3r.reshape(len(t3r),1)
            Linear_estimator3.fit(s3r, t3r)
            self.Linear_estimator3=Linear_estimator3
            
            
            
           
        if self.FLAG2==1:
               self.TwoClassModel=twoclass
        else :
               
                    
                d3=np.ones(len(t3))
                d4=np.ones(len(t4))*-1
                          
                X = pd.concat([s3,s4])
                y = np.concatenate([d3,d4],axis=0)
                self.y=y
                self.X=X
                #print (X)
                #self.decision_clf = svm.SVC()
                TwoClassModel = RandomForestClassifier()
                TwoClassModel.fit(X, y)
                self.TwoClassModel=TwoClassModel
                
            
        Linear_estimator4 = LinearRegression()        
        s4r=s4.full_sq
        s4r=s4r.reshape(len(s4r),1)
        t4r=t4
        t4r=t4r.reshape(len(t4r),1)
        Linear_estimator4.fit(s4r, t4r)
        self.Linear_estimator4=Linear_estimator4             

        
    def predict_price(self,test_row):
        testrow1=test_row
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
        if dum > self.highcutoff1:
            #predicted_price=5
            predicted_price1= self.random_estimator3.predict(test_row)[0] 
            
            x1 = test_row.full_sq
            x1 = x1.reshape(len(x1),1)
            
            predicted_price2= self.Linear_estimator3.predict(x1)[0]
            
        
            predicted_price=0.2*predicted_price1+ 0.8*predicted_price2
            
            
        else:
            if flag==1:
                
                predicted_price1= self.random_estimator2.predict(test_row)[0] 
                x1 = test_row.full_sq
                x1 = x1.reshape(len(x1),1)
                predicted_price2= self.Linear_estimator2.predict(x1)[0]
                predicted_price=0.5*predicted_price1+ 0.5*predicted_price2
            
            else:
                if self.FLAG2==1:
                    testrow1=testrow1.drop(['sub_area', 'full_sq' ,'product_type'],axis=1)
                    testrow1=testrow1.iloc[:,2:20]
                    for fea in testrow1.columns:
                              testrow1[fea].fillna(testrow1[fea].median(),inplace=True)
                    for fea in testrow1.columns:
                              testrow1[fea].fillna(0,inplace=True)
                              
                    pclass=self.TwoClassModel.predict(testrow1)[0]
                
                else:
                       pclass=self.TwoClassModel.predict(test_row)[0]
                       
                if pclass==-1:
                       #predicted_price= 1.5
                       x1 = test_row.full_sq
                       x1 = x1.reshape(len(x1),1)
                       predicted_price= self.Linear_estimator4.predict(x1)[0]
                else:                        
                       predicted_price1= self.random_estimator3.predict(test_row)[0] 
                       x1 = test_row.full_sq
                       x1 = x1.reshape(len(x1),1)
                       predicted_price2= self.Linear_estimator3.predict(x1)[0]
                       predicted_price=.5*predicted_price1+ .5*predicted_price2
                       
        if self.specialFlag==1:
            #if (dum >70) & (dum <600) & (predicted_price > 10):
            #     predicted_price=predicted_price + 3
            if (predicted_price <3) :
                 predicted_price +=dum/50
        #predicted_price -= .1
        return predicted_price 

gc.collect()
###############################################################
class Timirjazevskoe():
    #_data = pd.DataFrame
    def __init__(self, train_rows, twoclass):
        
        #### Flags
        self.specialFlag=0 ### this is for special adjustment in predict
        self.FLAG1=1  ### this is 1 for combined s2 s3
        self.FLAG2=0 ### this is 1 for global TwoClass classifier
        
        
        
        self.highcutoff = 150
        self.highcutoff1=150
        self.lowpricecutoff =3.8

        s1=train_rows[train_rows.full_sq > self.highcutoff];
        sa=train_rows[train_rows.full_sq < self.highcutoff];
        s2=sa[(sa.state.isnull()) & (sa.life_sq.isnull())]
        s3a = sa[~((sa.state.isnull()) & (sa.life_sq.isnull()))]
        s3=s3a[s3a.price_doc>self.lowpricecutoff]
        s4=s3a[s3a.price_doc <= self.lowpricecutoff]
        t1=s1.price_doc
        t2=s2.price_doc
        t3=s3.price_doc
        t4=s4.price_doc
        s1=s1.drop(['price_doc','sub_area','product_type'],axis=1)
        s1=s1.iloc[:,2:20]
        for fea in s1.columns:
                s1[fea].fillna(s1[fea].median(), inplace=True)
        for fea in s1.columns:
                s1[fea].fillna(0, inplace=True)
        s2=s2.drop(['price_doc','sub_area','product_type'],axis=1)
        s2=s2.iloc[:,2:20]
        for fea in s2.columns:
                s2[fea].fillna(s2[fea].median(), inplace=True)
        for fea in s2.columns:
                s2[fea].fillna(0, inplace=True)              
        s3=s3.drop(['price_doc','sub_area','product_type'],axis=1)
        s3=s3.iloc[:,2:20]
        for fea in s3.columns:
                s3[fea].fillna(s3[fea].median(), inplace=True)
        for fea in s3.columns:
                s3[fea].fillna(0, inplace=True)
        s4=s4.drop(['price_doc','sub_area','product_type'],axis=1)
        s4=s4.iloc[:,2:20]
        for fea in s4.columns:
                s4[fea].fillna(s4[fea].median(), inplace=True)
        for fea in s4.columns:
                s4[fea].fillna(0, inplace=True)
        
        sa=sa.drop(['price_doc','sub_area','product_type'],axis=1)
        sa=sa.iloc[:,2:20]
        for fea in sa.columns:
                sa[fea].fillna(sa[fea].median(), inplace=True)
        for fea in sa.columns:
                sa[fea].fillna(0, inplace=True)
                
                
        if self.FLAG1==1:
           X23 = pd.concat([s2,s3]) 
           y23 = np.concatenate([t2,t3],axis=0)
           random_estimator23=RandomForestRegressor(random_state=0, n_estimators=500)
           random_estimator23.fit(X23,y23)
           self.random_estimator2=random_estimator23
           self.random_estimator3=random_estimator23
           
           X23r=X23.full_sq
           X23r=X23r.reshape(len(X23r),1)
           y23r=y23.reshape(len(y23), 1)
           Linear_estimator23=LinearRegression()
           Linear_estimator23.fit(X23r, y23r)
           
           self.Linear_estimator2=Linear_estimator23
           self.Linear_estimator3=Linear_estimator23
           
        else:
            
            random_estimator2 = RandomForestRegressor(random_state=0, n_estimators=500)
            random_estimator2.fit(s2,t2)
            self.random_estimator2=random_estimator2
            
            Linear_estimator2 = LinearRegression()
            
            s2r=s2.full_sq
            s2r=s2r.reshape(len(s2r),1)
            t2r=t2
            t2r=t2r.reshape(len(t2r),1)
            Linear_estimator2.fit(s2r, t2r)
            self.Linear_estimator2=Linear_estimator2
            
            
            
            random_estimator3 = RandomForestRegressor(random_state=0, n_estimators=500)
            random_estimator3.fit(s3,t3)
            self.random_estimator3=random_estimator3
            
            Linear_estimator3 = LinearRegression()
            
            s3r=s3.full_sq
            s3r=s3r.reshape(len(s3r),1)
            t3r=t3
            t3r=t3r.reshape(len(t3r),1)
            Linear_estimator3.fit(s3r, t3r)
            self.Linear_estimator3=Linear_estimator3
            
            
            
           
        if self.FLAG2==1:
               self.TwoClassModel=twoclass
        else :
               
                    
                d3=np.ones(len(t3))
                d4=np.ones(len(t4))*-1
                          
                X = pd.concat([s3,s4])
                y = np.concatenate([d3,d4],axis=0)
                self.y=y
                self.X=X
                #print (X)
                #self.decision_clf = svm.SVC()
                TwoClassModel = RandomForestClassifier()
                TwoClassModel.fit(X, y)
                self.TwoClassModel=TwoClassModel
                
            
        Linear_estimator4 = LinearRegression()        
        s4r=s4.full_sq
        s4r=s4r.reshape(len(s4r),1)
        t4r=t4
        t4r=t4r.reshape(len(t4r),1)
        Linear_estimator4.fit(s4r, t4r)
        self.Linear_estimator4=Linear_estimator4             

        
    def predict_price(self,test_row):
        testrow1=test_row
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
        if dum > self.highcutoff1:
            #predicted_price=5
            predicted_price1= self.random_estimator3.predict(test_row)[0] 
            
            x1 = test_row.full_sq
            x1 = x1.reshape(len(x1),1)
            
            predicted_price2= self.Linear_estimator3.predict(x1)[0]
            
        
            predicted_price=0.2*predicted_price1+ 0.8*predicted_price2
            
            
        else:
            if flag==1:
                
                predicted_price1= self.random_estimator2.predict(test_row)[0] 
                x1 = test_row.full_sq
                x1 = x1.reshape(len(x1),1)
                predicted_price2= self.Linear_estimator2.predict(x1)[0]
                predicted_price=0.5*predicted_price1+ 0.5*predicted_price2
            
            else:
                if self.FLAG2==1:
                    testrow1=testrow1.drop(['sub_area', 'full_sq' ,'product_type'],axis=1)
                    testrow1=testrow1.iloc[:,2:20]
                    for fea in testrow1.columns:
                              testrow1[fea].fillna(testrow1[fea].median(),inplace=True)
                    for fea in testrow1.columns:
                              testrow1[fea].fillna(0,inplace=True)
                              
                    pclass=self.TwoClassModel.predict(testrow1)[0]
                
                else:
                       pclass=self.TwoClassModel.predict(test_row)[0]
                       
                if pclass==-1:
                       #predicted_price= 1.5
                       x1 = test_row.full_sq
                       x1 = x1.reshape(len(x1),1)
                       predicted_price= self.Linear_estimator4.predict(x1)[0]
                else:                        
                       predicted_price1= self.random_estimator3.predict(test_row)[0] 
                       x1 = test_row.full_sq
                       x1 = x1.reshape(len(x1),1)
                       predicted_price2= self.Linear_estimator3.predict(x1)[0]
                       predicted_price=.75*predicted_price1+ .25*predicted_price2
                       
        if self.specialFlag==1:
            #if (dum >70) & (dum <600) & (predicted_price > 10):
            #     predicted_price=predicted_price + 3
            if (predicted_price <3) :
                 predicted_price +=dum/50
        #predicted_price -= .1
        return predicted_price 

gc.collect()
###############################################################
class Jasenevo():
    #_data = pd.DataFrame
    def __init__(self, train_rows, twoclass):
        
        #### Flags
        self.specialFlag=0 ### this is for special adjustment in predict
        self.FLAG1=1  ### this is 1 for combined s2 s3
        self.FLAG2=0 ### this is 1 for global TwoClass classifier
        
        
        
        self.highcutoff = 150
        self.highcutoff1=150
        self.lowpricecutoff =4.8

        s1=train_rows[train_rows.full_sq > self.highcutoff];
        sa=train_rows[train_rows.full_sq < self.highcutoff];
        s2=sa[(sa.state.isnull()) & (sa.life_sq.isnull())]
        s3a = sa[~((sa.state.isnull()) & (sa.life_sq.isnull()))]
        s3=s3a[s3a.price_doc>self.lowpricecutoff]
        s4=s3a[s3a.price_doc <= self.lowpricecutoff]
        t1=s1.price_doc
        t2=s2.price_doc
        t3=s3.price_doc
        t4=s4.price_doc
        s1=s1.drop(['price_doc','sub_area','product_type'],axis=1)
        s1=s1.iloc[:,2:20]
        for fea in s1.columns:
                s1[fea].fillna(s1[fea].median(), inplace=True)
        for fea in s1.columns:
                s1[fea].fillna(0, inplace=True)
        s2=s2.drop(['price_doc','sub_area','product_type'],axis=1)
        s2=s2.iloc[:,2:20]
        for fea in s2.columns:
                s2[fea].fillna(s2[fea].median(), inplace=True)
        for fea in s2.columns:
                s2[fea].fillna(0, inplace=True)              
        s3=s3.drop(['price_doc','sub_area','product_type'],axis=1)
        s3=s3.iloc[:,2:20]
        for fea in s3.columns:
                s3[fea].fillna(s3[fea].median(), inplace=True)
        for fea in s3.columns:
                s3[fea].fillna(0, inplace=True)
        s4=s4.drop(['price_doc','sub_area','product_type'],axis=1)
        s4=s4.iloc[:,2:20]
        for fea in s4.columns:
                s4[fea].fillna(s4[fea].median(), inplace=True)
        for fea in s4.columns:
                s4[fea].fillna(0, inplace=True)
        
        sa=sa.drop(['price_doc','sub_area','product_type'],axis=1)
        sa=sa.iloc[:,2:20]
        for fea in sa.columns:
                sa[fea].fillna(sa[fea].median(), inplace=True)
        for fea in sa.columns:
                sa[fea].fillna(0, inplace=True)
                
                
        if self.FLAG1==1:
           X23 = pd.concat([s2,s3]) 
           y23 = np.concatenate([t2,t3],axis=0)
           random_estimator23=RandomForestRegressor(random_state=0, n_estimators=500)
           random_estimator23.fit(X23,y23)
           self.random_estimator2=random_estimator23
           self.random_estimator3=random_estimator23
           
           X23r=X23.full_sq
           X23r=X23r.reshape(len(X23r),1)
           y23r=y23.reshape(len(y23), 1)
           Linear_estimator23=LinearRegression()
           Linear_estimator23.fit(X23r, y23r)
           
           self.Linear_estimator2=Linear_estimator23
           self.Linear_estimator3=Linear_estimator23
           
        else:
            
            random_estimator2 = RandomForestRegressor(random_state=0, n_estimators=500)
            random_estimator2.fit(s2,t2)
            self.random_estimator2=random_estimator2
            
            Linear_estimator2 = LinearRegression()
            
            s2r=s2.full_sq
            s2r=s2r.reshape(len(s2r),1)
            t2r=t2
            t2r=t2r.reshape(len(t2r),1)
            Linear_estimator2.fit(s2r, t2r)
            self.Linear_estimator2=Linear_estimator2
            
            
            
            random_estimator3 = RandomForestRegressor(random_state=0, n_estimators=500)
            random_estimator3.fit(s3,t3)
            self.random_estimator3=random_estimator3
            
            Linear_estimator3 = LinearRegression()
            
            s3r=s3.full_sq
            s3r=s3r.reshape(len(s3r),1)
            t3r=t3
            t3r=t3r.reshape(len(t3r),1)
            Linear_estimator3.fit(s3r, t3r)
            self.Linear_estimator3=Linear_estimator3
            
            
            
           
        if self.FLAG2==1:
               self.TwoClassModel=twoclass
        else :
               
                    
                d3=np.ones(len(t3))
                d4=np.ones(len(t4))*-1
                          
                X = pd.concat([s3,s4])
                y = np.concatenate([d3,d4],axis=0)
                self.y=y
                self.X=X
                #print (X)
                #self.decision_clf = svm.SVC()
                TwoClassModel = RandomForestClassifier()
                TwoClassModel.fit(X, y)
                self.TwoClassModel=TwoClassModel
                
            
        Linear_estimator4 = LinearRegression()        
        s4r=s4.full_sq
        s4r=s4r.reshape(len(s4r),1)
        t4r=t4
        t4r=t4r.reshape(len(t4r),1)
        Linear_estimator4.fit(s4r, t4r)
        self.Linear_estimator4=Linear_estimator4             

        
    def predict_price(self,test_row):
        testrow1=test_row
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
        if dum > self.highcutoff1:
            #predicted_price=5
            predicted_price1= self.random_estimator3.predict(test_row)[0] 
            
            x1 = test_row.full_sq
            x1 = x1.reshape(len(x1),1)
            
            predicted_price2= self.Linear_estimator3.predict(x1)[0]
            
        
            predicted_price=0.2*predicted_price1+ 0.8*predicted_price2
            
            
        else:
            if flag==1:
                
                predicted_price1= self.random_estimator2.predict(test_row)[0] 
                x1 = test_row.full_sq
                x1 = x1.reshape(len(x1),1)
                predicted_price2= self.Linear_estimator2.predict(x1)[0]
                predicted_price=0.5*predicted_price1+ 0.5*predicted_price2
            
            else:
                if self.FLAG2==1:
                    testrow1=testrow1.drop(['sub_area', 'full_sq' ,'product_type'],axis=1)
                    testrow1=testrow1.iloc[:,2:20]
                    for fea in testrow1.columns:
                              testrow1[fea].fillna(testrow1[fea].median(),inplace=True)
                    for fea in testrow1.columns:
                              testrow1[fea].fillna(0,inplace=True)
                              
                    pclass=self.TwoClassModel.predict(testrow1)[0]
                
                else:
                       pclass=self.TwoClassModel.predict(test_row)[0]
                       
                if pclass==-1:
                       #predicted_price= 1.5
                       x1 = test_row.full_sq
                       x1 = x1.reshape(len(x1),1)
                       predicted_price= self.Linear_estimator4.predict(x1)[0]
                else:                        
                       predicted_price1= self.random_estimator3.predict(test_row)[0] 
                       x1 = test_row.full_sq
                       x1 = x1.reshape(len(x1),1)
                       predicted_price2= self.Linear_estimator3.predict(x1)[0]
                       predicted_price=.75*predicted_price1+ .25*predicted_price2
                       
        if self.specialFlag==1:
            #if (dum >70) & (dum <600) & (predicted_price > 10):
            #     predicted_price=predicted_price + 3
            if (dum > 180) :
                 predicted_price = 38
        #predicted_price -= .1
        return predicted_price 

gc.collect()
###############################################################
class Altufevskoe():
    #_data = pd.DataFrame
    def __init__(self, train_rows, twoclass):
        
        #### Flags
        self.specialFlag=0 ### this is for special adjustment in predict
        self.FLAG1=1  ### this is 1 for combined s2 s3
        self.FLAG2=0 ### this is 1 for global TwoClass classifier
        
        
        
        self.highcutoff = 150
        self.highcutoff1=150
        self.lowpricecutoff =3.2

        s1=train_rows[train_rows.full_sq > self.highcutoff];
        sa=train_rows[train_rows.full_sq < self.highcutoff];
        s2=sa[(sa.state.isnull()) & (sa.life_sq.isnull())]
        s3a = sa[~((sa.state.isnull()) & (sa.life_sq.isnull()))]
        s3=s3a[s3a.price_doc>self.lowpricecutoff]
        s4=s3a[s3a.price_doc <= self.lowpricecutoff]
        t1=s1.price_doc
        t2=s2.price_doc
        t3=s3.price_doc
        t4=s4.price_doc
        s1=s1.drop(['price_doc','sub_area','product_type'],axis=1)
        s1=s1.iloc[:,2:20]
        for fea in s1.columns:
                s1[fea].fillna(s1[fea].median(), inplace=True)
        for fea in s1.columns:
                s1[fea].fillna(0, inplace=True)
        s2=s2.drop(['price_doc','sub_area','product_type'],axis=1)
        s2=s2.iloc[:,2:20]
        for fea in s2.columns:
                s2[fea].fillna(s2[fea].median(), inplace=True)
        for fea in s2.columns:
                s2[fea].fillna(0, inplace=True)              
        s3=s3.drop(['price_doc','sub_area','product_type'],axis=1)
        s3=s3.iloc[:,2:20]
        for fea in s3.columns:
                s3[fea].fillna(s3[fea].median(), inplace=True)
        for fea in s3.columns:
                s3[fea].fillna(0, inplace=True)
        s4=s4.drop(['price_doc','sub_area','product_type'],axis=1)
        s4=s4.iloc[:,2:20]
        for fea in s4.columns:
                s4[fea].fillna(s4[fea].median(), inplace=True)
        for fea in s4.columns:
                s4[fea].fillna(0, inplace=True)
        
        sa=sa.drop(['price_doc','sub_area','product_type'],axis=1)
        sa=sa.iloc[:,2:20]
        for fea in sa.columns:
                sa[fea].fillna(sa[fea].median(), inplace=True)
        for fea in sa.columns:
                sa[fea].fillna(0, inplace=True)
                
                
        if self.FLAG1==1:
           X23 = pd.concat([s2,s3]) 
           y23 = np.concatenate([t2,t3],axis=0)
           random_estimator23=RandomForestRegressor(random_state=0, n_estimators=500)
           random_estimator23.fit(X23,y23)
           self.random_estimator2=random_estimator23
           self.random_estimator3=random_estimator23
           
           X23r=X23.full_sq
           X23r=X23r.reshape(len(X23r),1)
           y23r=y23.reshape(len(y23), 1)
           Linear_estimator23=LinearRegression()
           Linear_estimator23.fit(X23r, y23r)
           
           self.Linear_estimator2=Linear_estimator23
           self.Linear_estimator3=Linear_estimator23
           
        else:
            
            random_estimator2 = RandomForestRegressor(random_state=0, n_estimators=500)
            random_estimator2.fit(s2,t2)
            self.random_estimator2=random_estimator2
            
            Linear_estimator2 = LinearRegression()
            
            s2r=s2.full_sq
            s2r=s2r.reshape(len(s2r),1)
            t2r=t2
            t2r=t2r.reshape(len(t2r),1)
            Linear_estimator2.fit(s2r, t2r)
            self.Linear_estimator2=Linear_estimator2
            
            
            
            random_estimator3 = RandomForestRegressor(random_state=0, n_estimators=500)
            random_estimator3.fit(s3,t3)
            self.random_estimator3=random_estimator3
            
            Linear_estimator3 = LinearRegression()
            
            s3r=s3.full_sq
            s3r=s3r.reshape(len(s3r),1)
            t3r=t3
            t3r=t3r.reshape(len(t3r),1)
            Linear_estimator3.fit(s3r, t3r)
            self.Linear_estimator3=Linear_estimator3
            
            
            
           
        if self.FLAG2==1:
               self.TwoClassModel=twoclass
        else :
               
                    
                d3=np.ones(len(t3))
                d4=np.ones(len(t4))*-1
                          
                X = pd.concat([s3,s4])
                y = np.concatenate([d3,d4],axis=0)
                self.y=y
                self.X=X
                #print (X)
                #self.decision_clf = svm.SVC()
                TwoClassModel = RandomForestClassifier()
                TwoClassModel.fit(X, y)
                self.TwoClassModel=TwoClassModel
                
            
        Linear_estimator4 = LinearRegression()        
        s4r=s4.full_sq
        s4r=s4r.reshape(len(s4r),1)
        t4r=t4
        t4r=t4r.reshape(len(t4r),1)
        Linear_estimator4.fit(s4r, t4r)
        self.Linear_estimator4=Linear_estimator4             

        
    def predict_price(self,test_row):
        testrow1=test_row
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
        if dum > self.highcutoff1:
            #predicted_price=5
            predicted_price1= self.random_estimator3.predict(test_row)[0] 
            
            x1 = test_row.full_sq
            x1 = x1.reshape(len(x1),1)
            
            predicted_price2= self.Linear_estimator3.predict(x1)[0]
            
        
            predicted_price=0.2*predicted_price1+ 0.8*predicted_price2
            
            
        else:
            if flag==1:
                
                predicted_price1= self.random_estimator2.predict(test_row)[0] 
                x1 = test_row.full_sq
                x1 = x1.reshape(len(x1),1)
                predicted_price2= self.Linear_estimator2.predict(x1)[0]
                predicted_price=0.5*predicted_price1+ 0.5*predicted_price2
            
            else:
                if self.FLAG2==1:
                    testrow1=testrow1.drop(['sub_area', 'full_sq' ,'product_type'],axis=1)
                    testrow1=testrow1.iloc[:,2:20]
                    for fea in testrow1.columns:
                              testrow1[fea].fillna(testrow1[fea].median(),inplace=True)
                    for fea in testrow1.columns:
                              testrow1[fea].fillna(0,inplace=True)
                              
                    pclass=self.TwoClassModel.predict(testrow1)[0]
                
                else:
                       pclass=self.TwoClassModel.predict(test_row)[0]
                       
                if pclass==-1:
                       #predicted_price= 1.5
                       x1 = test_row.full_sq
                       x1 = x1.reshape(len(x1),1)
                       predicted_price= self.Linear_estimator4.predict(x1)[0]
                else:                        
                       predicted_price1= self.random_estimator3.predict(test_row)[0] 
                       x1 = test_row.full_sq
                       x1 = x1.reshape(len(x1),1)
                       predicted_price2= self.Linear_estimator3.predict(x1)[0]
                       predicted_price=.75*predicted_price1+ .25*predicted_price2
                       
        if self.specialFlag==1:
            #if (dum >70) & (dum <600) & (predicted_price > 10):
            #     predicted_price=predicted_price + 3
            if (dum > 180) :
                 predicted_price = 38
        #predicted_price -= .1
        return predicted_price 






################################################################################################################
################################################################################################################
################################################################################################################
################################################################################################################
################################################################################################################
################################################################################################################
################################################################################################################
################################################################################################################
################################################################################################################
################################################################################################################
################################################################################################################
################################################################################################################
################################################################################################################
################################################################################################################
################################################################################################################
################################################################################################################
################################################################################################################
################################################################################################################
################################################################################################################
################################################################################################################
################################################################################################################
################################################################################################################
class Severnoe_Medvedkovo():
    #_data = pd.DataFrame
    def __init__(self, train_rows, twoclass):
        
        #### Flags
        self.specialFlag=0 ### this is for special adjustment in predict
        self.FLAG1=1  ### this is 1 for combined s2 s3
        self.FLAG2=0 ### this is 1 for global TwoClass classifier
        
        
        
        self.highcutoff = 150
        self.highcutoff1=150
        self.lowpricecutoff =3

        s1=train_rows[train_rows.full_sq > self.highcutoff];
        sa=train_rows[train_rows.full_sq < self.highcutoff];
        s2=sa[(sa.state.isnull()) & (sa.life_sq.isnull())]
        s3a = sa[~((sa.state.isnull()) & (sa.life_sq.isnull()))]
        s3=s3a[s3a.price_doc>self.lowpricecutoff]
        s4=s3a[s3a.price_doc <= self.lowpricecutoff]
        t1=s1.price_doc
        t2=s2.price_doc
        t3=s3.price_doc
        t4=s4.price_doc
        s1=s1.drop(['price_doc','sub_area','product_type'],axis=1)
        s1=s1.iloc[:,2:20]
        for fea in s1.columns:
                s1[fea].fillna(s1[fea].median(), inplace=True)
        for fea in s1.columns:
                s1[fea].fillna(0, inplace=True)
        s2=s2.drop(['price_doc','sub_area','product_type'],axis=1)
        s2=s2.iloc[:,2:20]
        for fea in s2.columns:
                s2[fea].fillna(s2[fea].median(), inplace=True)
        for fea in s2.columns:
                s2[fea].fillna(0, inplace=True)              
        s3=s3.drop(['price_doc','sub_area','product_type'],axis=1)
        s3=s3.iloc[:,2:20]
        for fea in s3.columns:
                s3[fea].fillna(s3[fea].median(), inplace=True)
        for fea in s3.columns:
                s3[fea].fillna(0, inplace=True)
        s4=s4.drop(['price_doc','sub_area','product_type'],axis=1)
        s4=s4.iloc[:,2:20]
        for fea in s4.columns:
                s4[fea].fillna(s4[fea].median(), inplace=True)
        for fea in s4.columns:
                s4[fea].fillna(0, inplace=True)
        
        sa=sa.drop(['price_doc','sub_area','product_type'],axis=1)
        sa=sa.iloc[:,2:20]
        for fea in sa.columns:
                sa[fea].fillna(sa[fea].median(), inplace=True)
        for fea in sa.columns:
                sa[fea].fillna(0, inplace=True)
                
                
        if self.FLAG1==1:
           X23 = pd.concat([s2,s3]) 
           y23 = np.concatenate([t2,t3],axis=0)
           random_estimator23=RandomForestRegressor(random_state=0, n_estimators=500)
           random_estimator23.fit(X23,y23)
           self.random_estimator2=random_estimator23
           self.random_estimator3=random_estimator23
           
           X23r=X23.full_sq
           X23r=X23r.reshape(len(X23r),1)
           y23r=y23.reshape(len(y23), 1)
           Linear_estimator23=LinearRegression()
           Linear_estimator23.fit(X23r, y23r)
           
           self.Linear_estimator2=Linear_estimator23
           self.Linear_estimator3=Linear_estimator23
           
        else:
            
            random_estimator2 = RandomForestRegressor(random_state=0, n_estimators=500)
            random_estimator2.fit(s2,t2)
            self.random_estimator2=random_estimator2
            
            Linear_estimator2 = LinearRegression()
            
            s2r=s2.full_sq
            s2r=s2r.reshape(len(s2r),1)
            t2r=t2
            t2r=t2r.reshape(len(t2r),1)
            Linear_estimator2.fit(s2r, t2r)
            self.Linear_estimator2=Linear_estimator2
            
            
            
            random_estimator3 = RandomForestRegressor(random_state=0, n_estimators=500)
            random_estimator3.fit(s3,t3)
            self.random_estimator3=random_estimator3
            
            Linear_estimator3 = LinearRegression()
            
            s3r=s3.full_sq
            s3r=s3r.reshape(len(s3r),1)
            t3r=t3
            t3r=t3r.reshape(len(t3r),1)
            Linear_estimator3.fit(s3r, t3r)
            self.Linear_estimator3=Linear_estimator3
            
            
            
           
        if self.FLAG2==1:
               self.TwoClassModel=twoclass
        else :
               
                    
                d3=np.ones(len(t3))
                d4=np.ones(len(t4))*-1
                          
                X = pd.concat([s3,s4])
                y = np.concatenate([d3,d4],axis=0)
                self.y=y
                self.X=X
                #print (X)
                #self.decision_clf = svm.SVC()
                TwoClassModel = RandomForestClassifier()
                TwoClassModel.fit(X, y)
                self.TwoClassModel=TwoClassModel
                
            
        Linear_estimator4 = LinearRegression()        
        s4r=s4.full_sq
        s4r=s4r.reshape(len(s4r),1)
        t4r=t4
        t4r=t4r.reshape(len(t4r),1)
        Linear_estimator4.fit(s4r, t4r)
        self.Linear_estimator4=Linear_estimator4             

        
    def predict_price(self,test_row):
        testrow1=test_row
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
        if dum > self.highcutoff1:
            #predicted_price=5
            predicted_price1= self.random_estimator3.predict(test_row)[0] 
            
            x1 = test_row.full_sq
            x1 = x1.reshape(len(x1),1)
            
            predicted_price2= self.Linear_estimator3.predict(x1)[0]
            
        
            predicted_price=0.2*predicted_price1+ 0.8*predicted_price2
            
            
        else:
            if flag==1:
                
                predicted_price1= self.random_estimator2.predict(test_row)[0] 
                x1 = test_row.full_sq
                x1 = x1.reshape(len(x1),1)
                predicted_price2= self.Linear_estimator2.predict(x1)[0]
                predicted_price=0.5*predicted_price1+ 0.5*predicted_price2
            
            else:
                if self.FLAG2==1:
                    testrow1=testrow1.drop(['sub_area', 'full_sq' ,'product_type'],axis=1)
                    testrow1=testrow1.iloc[:,2:20]
                    for fea in testrow1.columns:
                              testrow1[fea].fillna(testrow1[fea].median(),inplace=True)
                    for fea in testrow1.columns:
                              testrow1[fea].fillna(0,inplace=True)
                              
                    pclass=self.TwoClassModel.predict(testrow1)[0]
                
                else:
                       pclass=self.TwoClassModel.predict(test_row)[0]
                       
                if pclass==-1:
                       #predicted_price= 1.5
                       x1 = test_row.full_sq
                       x1 = x1.reshape(len(x1),1)
                       predicted_price= self.Linear_estimator4.predict(x1)[0]
                else:                        
                       predicted_price1= self.random_estimator3.predict(test_row)[0] 
                       x1 = test_row.full_sq
                       x1 = x1.reshape(len(x1),1)
                       predicted_price2= self.Linear_estimator3.predict(x1)[0]
                       predicted_price=.75*predicted_price1+ .25*predicted_price2
                       
        if self.specialFlag==1:
            #if (dum >70) & (dum <600) & (predicted_price > 10):
            #     predicted_price=predicted_price + 3
            if (dum > 180) :
                 predicted_price = 38
        #predicted_price -= .1
        return predicted_price 
class Kurkino():
    #_data = pd.DataFrame
    def __init__(self, train_rows, twoclass):
        
        #### Flags
        self.specialFlag=0 ### this is for special adjustment in predict
        self.FLAG1=1  ### this is 1 for combined s2 s3
        self.FLAG2=0 ### this is 1 for global TwoClass classifier
        
        
        
        self.highcutoff = 150
        self.highcutoff1=150
        self.lowpricecutoff =7

        s1=train_rows[train_rows.full_sq > self.highcutoff];
        sa=train_rows[train_rows.full_sq < self.highcutoff];
        s2=sa[(sa.state.isnull()) & (sa.life_sq.isnull())]
        s3a = sa[~((sa.state.isnull()) & (sa.life_sq.isnull()))]
        s3=s3a[s3a.price_doc>self.lowpricecutoff]
        s4=s3a[s3a.price_doc <= self.lowpricecutoff]
        t1=s1.price_doc
        t2=s2.price_doc
        t3=s3.price_doc
        t4=s4.price_doc
        s1=s1.drop(['price_doc','sub_area','product_type'],axis=1)
        s1=s1.iloc[:,2:20]
        for fea in s1.columns:
                s1[fea].fillna(s1[fea].median(), inplace=True)
        for fea in s1.columns:
                s1[fea].fillna(0, inplace=True)
        s2=s2.drop(['price_doc','sub_area','product_type'],axis=1)
        s2=s2.iloc[:,2:20]
        for fea in s2.columns:
                s2[fea].fillna(s2[fea].median(), inplace=True)
        for fea in s2.columns:
                s2[fea].fillna(0, inplace=True)              
        s3=s3.drop(['price_doc','sub_area','product_type'],axis=1)
        s3=s3.iloc[:,2:20]
        for fea in s3.columns:
                s3[fea].fillna(s3[fea].median(), inplace=True)
        for fea in s3.columns:
                s3[fea].fillna(0, inplace=True)
        s4=s4.drop(['price_doc','sub_area','product_type'],axis=1)
        s4=s4.iloc[:,2:20]
        for fea in s4.columns:
                s4[fea].fillna(s4[fea].median(), inplace=True)
        for fea in s4.columns:
                s4[fea].fillna(0, inplace=True)
        
        sa=sa.drop(['price_doc','sub_area','product_type'],axis=1)
        sa=sa.iloc[:,2:20]
        for fea in sa.columns:
                sa[fea].fillna(sa[fea].median(), inplace=True)
        for fea in sa.columns:
                sa[fea].fillna(0, inplace=True)
                
                
        if self.FLAG1==1:
           X23 = pd.concat([s2,s3]) 
           y23 = np.concatenate([t2,t3],axis=0)
           random_estimator23=RandomForestRegressor(random_state=0, n_estimators=500)
           random_estimator23.fit(X23,y23)
           self.random_estimator2=random_estimator23
           self.random_estimator3=random_estimator23
           
           X23r=X23.full_sq
           X23r=X23r.reshape(len(X23r),1)
           y23r=y23.reshape(len(y23), 1)
           Linear_estimator23=LinearRegression()
           Linear_estimator23.fit(X23r, y23r)
           
           self.Linear_estimator2=Linear_estimator23
           self.Linear_estimator3=Linear_estimator23
           
        else:
            
            random_estimator2 = RandomForestRegressor(random_state=0, n_estimators=500)
            random_estimator2.fit(s2,t2)
            self.random_estimator2=random_estimator2
            
            Linear_estimator2 = LinearRegression()
            
            s2r=s2.full_sq
            s2r=s2r.reshape(len(s2r),1)
            t2r=t2
            t2r=t2r.reshape(len(t2r),1)
            Linear_estimator2.fit(s2r, t2r)
            self.Linear_estimator2=Linear_estimator2
            
            
            
            random_estimator3 = RandomForestRegressor(random_state=0, n_estimators=500)
            random_estimator3.fit(s3,t3)
            self.random_estimator3=random_estimator3
            
            Linear_estimator3 = LinearRegression()
            
            s3r=s3.full_sq
            s3r=s3r.reshape(len(s3r),1)
            t3r=t3
            t3r=t3r.reshape(len(t3r),1)
            Linear_estimator3.fit(s3r, t3r)
            self.Linear_estimator3=Linear_estimator3
            
            
            
           
        if self.FLAG2==1:
               self.TwoClassModel=twoclass
        else :
               
                    
                d3=np.ones(len(t3))
                d4=np.ones(len(t4))*-1
                          
                X = pd.concat([s3,s4])
                y = np.concatenate([d3,d4],axis=0)
                self.y=y
                self.X=X
                #print (X)
                #self.decision_clf = svm.SVC()
                TwoClassModel = RandomForestClassifier()
                TwoClassModel.fit(X, y)
                self.TwoClassModel=TwoClassModel
                
            
        Linear_estimator4 = LinearRegression()        
        s4r=s4.full_sq
        s4r=s4r.reshape(len(s4r),1)
        t4r=t4
        t4r=t4r.reshape(len(t4r),1)
        Linear_estimator4.fit(s4r, t4r)
        self.Linear_estimator4=Linear_estimator4             

        
    def predict_price(self,test_row):
        testrow1=test_row
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
        if dum > self.highcutoff1:
            #predicted_price=5
            predicted_price1= self.random_estimator3.predict(test_row)[0] 
            
            x1 = test_row.full_sq
            x1 = x1.reshape(len(x1),1)
            
            predicted_price2= self.Linear_estimator3.predict(x1)[0]
            
        
            predicted_price=0.2*predicted_price1+ 0.8*predicted_price2
            
            
        else:
            if flag==1:
                
                predicted_price1= self.random_estimator2.predict(test_row)[0] 
                x1 = test_row.full_sq
                x1 = x1.reshape(len(x1),1)
                predicted_price2= self.Linear_estimator2.predict(x1)[0]
                predicted_price=0.5*predicted_price1+ 0.5*predicted_price2
            
            else:
                if self.FLAG2==1:
                    testrow1=testrow1.drop(['sub_area', 'full_sq' ,'product_type'],axis=1)
                    testrow1=testrow1.iloc[:,2:20]
                    for fea in testrow1.columns:
                              testrow1[fea].fillna(testrow1[fea].median(),inplace=True)
                    for fea in testrow1.columns:
                              testrow1[fea].fillna(0,inplace=True)
                              
                    pclass=self.TwoClassModel.predict(testrow1)[0]
                
                else:
                       pclass=self.TwoClassModel.predict(test_row)[0]
                       
                if pclass==-1:
                       #predicted_price= 1.5
                       x1 = test_row.full_sq
                       x1 = x1.reshape(len(x1),1)
                       predicted_price= self.Linear_estimator4.predict(x1)[0]
                else:                        
                       predicted_price1= self.random_estimator3.predict(test_row)[0] 
                       x1 = test_row.full_sq
                       x1 = x1.reshape(len(x1),1)
                       predicted_price2= self.Linear_estimator3.predict(x1)[0]
                       predicted_price=.75*predicted_price1+ .25*predicted_price2
                       
        if self.specialFlag==1:
            #if (dum >70) & (dum <600) & (predicted_price > 10):
            #     predicted_price=predicted_price + 3
            if (dum > 180) :
                 predicted_price = 38
        #predicted_price -= .1
        return predicted_price 

class Vyhino_Zhulebino():
    #_data = pd.DataFrame
    def __init__(self, train_rows, twoclass):
        
        #### Flags
        self.specialFlag=0 ### this is for special adjustment in predict
        self.FLAG1=1  ### this is 1 for combined s2 s3
        self.FLAG2=0 ### this is 1 for global TwoClass classifier
        
        
        
        self.highcutoff = 150
        self.highcutoff1=150
        self.lowpricecutoff =4.2

        s1=train_rows[train_rows.full_sq > self.highcutoff];
        sa=train_rows[train_rows.full_sq < self.highcutoff];
        s2=sa[(sa.state.isnull()) & (sa.life_sq.isnull())]
        s3a = sa[~((sa.state.isnull()) & (sa.life_sq.isnull()))]
        s3=s3a[s3a.price_doc>self.lowpricecutoff]
        s4=s3a[s3a.price_doc <= self.lowpricecutoff]
        t1=s1.price_doc
        t2=s2.price_doc
        t3=s3.price_doc
        t4=s4.price_doc
        s1=s1.drop(['price_doc','sub_area','product_type'],axis=1)
        s1=s1.iloc[:,2:20]
        for fea in s1.columns:
                s1[fea].fillna(s1[fea].median(), inplace=True)
        for fea in s1.columns:
                s1[fea].fillna(0, inplace=True)
        s2=s2.drop(['price_doc','sub_area','product_type'],axis=1)
        s2=s2.iloc[:,2:20]
        for fea in s2.columns:
                s2[fea].fillna(s2[fea].median(), inplace=True)
        for fea in s2.columns:
                s2[fea].fillna(0, inplace=True)              
        s3=s3.drop(['price_doc','sub_area','product_type'],axis=1)
        s3=s3.iloc[:,2:20]
        for fea in s3.columns:
                s3[fea].fillna(s3[fea].median(), inplace=True)
        for fea in s3.columns:
                s3[fea].fillna(0, inplace=True)
        s4=s4.drop(['price_doc','sub_area','product_type'],axis=1)
        s4=s4.iloc[:,2:20]
        for fea in s4.columns:
                s4[fea].fillna(s4[fea].median(), inplace=True)
        for fea in s4.columns:
                s4[fea].fillna(0, inplace=True)
        
        sa=sa.drop(['price_doc','sub_area','product_type'],axis=1)
        sa=sa.iloc[:,2:20]
        for fea in sa.columns:
                sa[fea].fillna(sa[fea].median(), inplace=True)
        for fea in sa.columns:
                sa[fea].fillna(0, inplace=True)
                
                
        if self.FLAG1==1:
           X23 = pd.concat([s2,s3]) 
           y23 = np.concatenate([t2,t3],axis=0)
           random_estimator23=RandomForestRegressor(random_state=0, n_estimators=500)
           random_estimator23.fit(X23,y23)
           self.random_estimator2=random_estimator23
           self.random_estimator3=random_estimator23
           
           X23r=X23.full_sq
           X23r=X23r.reshape(len(X23r),1)
           y23r=y23.reshape(len(y23), 1)
           Linear_estimator23=LinearRegression()
           Linear_estimator23.fit(X23r, y23r)
           
           self.Linear_estimator2=Linear_estimator23
           self.Linear_estimator3=Linear_estimator23
           
        else:
            
            random_estimator2 = RandomForestRegressor(random_state=0, n_estimators=500)
            random_estimator2.fit(s2,t2)
            self.random_estimator2=random_estimator2
            
            Linear_estimator2 = LinearRegression()
            
            s2r=s2.full_sq
            s2r=s2r.reshape(len(s2r),1)
            t2r=t2
            t2r=t2r.reshape(len(t2r),1)
            Linear_estimator2.fit(s2r, t2r)
            self.Linear_estimator2=Linear_estimator2
            
            
            
            random_estimator3 = RandomForestRegressor(random_state=0, n_estimators=500)
            random_estimator3.fit(s3,t3)
            self.random_estimator3=random_estimator3
            
            Linear_estimator3 = LinearRegression()
            
            s3r=s3.full_sq
            s3r=s3r.reshape(len(s3r),1)
            t3r=t3
            t3r=t3r.reshape(len(t3r),1)
            Linear_estimator3.fit(s3r, t3r)
            self.Linear_estimator3=Linear_estimator3
            
            
            
           
        if self.FLAG2==1:
               self.TwoClassModel=twoclass
        else :
               
                    
                d3=np.ones(len(t3))
                d4=np.ones(len(t4))*-1
                          
                X = pd.concat([s3,s4])
                y = np.concatenate([d3,d4],axis=0)
                self.y=y
                self.X=X
                #print (X)
                #self.decision_clf = svm.SVC()
                TwoClassModel = RandomForestClassifier()
                TwoClassModel.fit(X, y)
                self.TwoClassModel=TwoClassModel
                
            
        Linear_estimator4 = LinearRegression()        
        s4r=s4.full_sq
        s4r=s4r.reshape(len(s4r),1)
        t4r=t4
        t4r=t4r.reshape(len(t4r),1)
        Linear_estimator4.fit(s4r, t4r)
        self.Linear_estimator4=Linear_estimator4             

        
    def predict_price(self,test_row):
        testrow1=test_row
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
        if dum > self.highcutoff1:
            #predicted_price=5
            predicted_price1= self.random_estimator3.predict(test_row)[0] 
            
            x1 = test_row.full_sq
            x1 = x1.reshape(len(x1),1)
            
            predicted_price2= self.Linear_estimator3.predict(x1)[0]
            
        
            predicted_price=0.2*predicted_price1+ 0.8*predicted_price2
            
            
        else:
            if flag==1:
                
                predicted_price1= self.random_estimator2.predict(test_row)[0] 
                x1 = test_row.full_sq
                x1 = x1.reshape(len(x1),1)
                predicted_price2= self.Linear_estimator2.predict(x1)[0]
                predicted_price=0.5*predicted_price1+ 0.5*predicted_price2
            
            else:
                if self.FLAG2==1:
                    testrow1=testrow1.drop(['sub_area', 'full_sq' ,'product_type'],axis=1)
                    testrow1=testrow1.iloc[:,2:20]
                    for fea in testrow1.columns:
                              testrow1[fea].fillna(testrow1[fea].median(),inplace=True)
                    for fea in testrow1.columns:
                              testrow1[fea].fillna(0,inplace=True)
                              
                    pclass=self.TwoClassModel.predict(testrow1)[0]
                
                else:
                       pclass=self.TwoClassModel.predict(test_row)[0]
                       
                if pclass==-1:
                       #predicted_price= 1.5
                       x1 = test_row.full_sq
                       x1 = x1.reshape(len(x1),1)
                       predicted_price= self.Linear_estimator4.predict(x1)[0]
                else:                        
                       predicted_price1= self.random_estimator3.predict(test_row)[0] 
                       x1 = test_row.full_sq
                       x1 = x1.reshape(len(x1),1)
                       predicted_price2= self.Linear_estimator3.predict(x1)[0]
                       predicted_price=.75*predicted_price1+ .25*predicted_price2
                       
        if self.specialFlag==1:
            #if (dum >70) & (dum <600) & (predicted_price > 10):
            #     predicted_price=predicted_price + 3
            if (dum > 180) :
                 predicted_price = 38
        #predicted_price -= .1
        return predicted_price 

class Filevskij_Park():
    #_data = pd.DataFrame
    def __init__(self, train_rows, twoclass):
        
        #### Flags
        self.specialFlag=1 ### this is for special adjustment in predict
        self.FLAG1=1  ### this is 1 for combined s2 s3
        self.FLAG2=1 ### this is 1 for global TwoClass classifier
        
        
        
        self.highcutoff = 150
        self.highcutoff1=150
        self.lowpricecutoff =5.3

        s1=train_rows[train_rows.full_sq > self.highcutoff];
        sa=train_rows[train_rows.full_sq < self.highcutoff];
        s2=sa[(sa.state.isnull()) & (sa.life_sq.isnull())]
        s3a = sa[~((sa.state.isnull()) & (sa.life_sq.isnull()))]
        s3=s3a[s3a.price_doc>self.lowpricecutoff]
        s4=s3a[s3a.price_doc <= self.lowpricecutoff]
        t1=s1.price_doc
        t2=s2.price_doc
        t3=s3.price_doc
        t4=s4.price_doc
        s1=s1.drop(['price_doc','sub_area','product_type'],axis=1)
        s1=s1.iloc[:,2:20]
        for fea in s1.columns:
                s1[fea].fillna(s1[fea].median(), inplace=True)
        for fea in s1.columns:
                s1[fea].fillna(0, inplace=True)
        s2=s2.drop(['price_doc','sub_area','product_type'],axis=1)
        s2=s2.iloc[:,2:20]
        for fea in s2.columns:
                s2[fea].fillna(s2[fea].median(), inplace=True)
        for fea in s2.columns:
                s2[fea].fillna(0, inplace=True)              
        s3=s3.drop(['price_doc','sub_area','product_type'],axis=1)
        s3=s3.iloc[:,2:20]
        for fea in s3.columns:
                s3[fea].fillna(s3[fea].median(), inplace=True)
        for fea in s3.columns:
                s3[fea].fillna(0, inplace=True)
        s4=s4.drop(['price_doc','sub_area','product_type'],axis=1)
        s4=s4.iloc[:,2:20]
        for fea in s4.columns:
                s4[fea].fillna(s4[fea].median(), inplace=True)
        for fea in s4.columns:
                s4[fea].fillna(0, inplace=True)
        
        sa=sa.drop(['price_doc','sub_area','product_type'],axis=1)
        sa=sa.iloc[:,2:20]
        for fea in sa.columns:
                sa[fea].fillna(sa[fea].median(), inplace=True)
        for fea in sa.columns:
                sa[fea].fillna(0, inplace=True)
                
                
        if self.FLAG1==1:
           X23 = pd.concat([s2,s3]) 
           y23 = np.concatenate([t2,t3],axis=0)
           random_estimator23=RandomForestRegressor(random_state=0, n_estimators=500)
           random_estimator23.fit(X23,y23)
           self.random_estimator2=random_estimator23
           self.random_estimator3=random_estimator23
           
           X23r=X23.full_sq
           X23r=X23r.reshape(len(X23r),1)
           y23r=y23.reshape(len(y23), 1)
           Linear_estimator23=LinearRegression()
           Linear_estimator23.fit(X23r, y23r)
           
           self.Linear_estimator2=Linear_estimator23
           self.Linear_estimator3=Linear_estimator23
           
        else:
            
            random_estimator2 = RandomForestRegressor(random_state=0, n_estimators=500)
            random_estimator2.fit(s2,t2)
            self.random_estimator2=random_estimator2
            
            Linear_estimator2 = LinearRegression()
            
            s2r=s2.full_sq
            s2r=s2r.reshape(len(s2r),1)
            t2r=t2
            t2r=t2r.reshape(len(t2r),1)
            Linear_estimator2.fit(s2r, t2r)
            self.Linear_estimator2=Linear_estimator2
            
            
            
            random_estimator3 = RandomForestRegressor(random_state=0, n_estimators=500)
            random_estimator3.fit(s3,t3)
            self.random_estimator3=random_estimator3
            
            Linear_estimator3 = LinearRegression()
            
            s3r=s3.full_sq
            s3r=s3r.reshape(len(s3r),1)
            t3r=t3
            t3r=t3r.reshape(len(t3r),1)
            Linear_estimator3.fit(s3r, t3r)
            self.Linear_estimator3=Linear_estimator3
            
            
            
           
        if self.FLAG2==1:
               self.TwoClassModel=twoclass
        else :
               
                    
                d3=np.ones(len(t3))
                d4=np.ones(len(t4))*-1
                          
                X = pd.concat([s3,s4])
                y = np.concatenate([d3,d4],axis=0)
                self.y=y
                self.X=X
                #print (X)
                #self.decision_clf = svm.SVC()
                TwoClassModel = RandomForestClassifier()
                TwoClassModel.fit(X, y)
                self.TwoClassModel=TwoClassModel
                
            
        Linear_estimator4 = LinearRegression()        
        s4r=s4.full_sq
        s4r=s4r.reshape(len(s4r),1)
        t4r=t4
        t4r=t4r.reshape(len(t4r),1)
        Linear_estimator4.fit(s4r, t4r)
        self.Linear_estimator4=Linear_estimator4             

        
    def predict_price(self,test_row):
        testrow1=test_row
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
        if dum > self.highcutoff1:
            #predicted_price=5
            predicted_price1= self.random_estimator3.predict(test_row)[0] 
            
            x1 = test_row.full_sq
            x1 = x1.reshape(len(x1),1)
            
            predicted_price2= self.Linear_estimator3.predict(x1)[0]
            
        
            predicted_price=0.2*predicted_price1+ 0.8*predicted_price2
            
            
        else:
            if flag==1:
                
                predicted_price1= self.random_estimator2.predict(test_row)[0] 
                x1 = test_row.full_sq
                x1 = x1.reshape(len(x1),1)
                predicted_price2= self.Linear_estimator2.predict(x1)[0]
                predicted_price=0.5*predicted_price1+ 0.5*predicted_price2
            
            else:
                if self.FLAG2==1:
                    testrow1=testrow1.drop(['sub_area', 'full_sq' ,'product_type'],axis=1)
                    testrow1=testrow1.iloc[:,2:20]
                    for fea in testrow1.columns:
                              testrow1[fea].fillna(testrow1[fea].median(),inplace=True)
                    for fea in testrow1.columns:
                              testrow1[fea].fillna(0,inplace=True)
                              
                    pclass=self.TwoClassModel.predict(testrow1)[0]
                
                else:
                       pclass=self.TwoClassModel.predict(test_row)[0]
                       
                if pclass==-1:
                       #predicted_price= 1.5
                       x1 = test_row.full_sq
                       x1 = x1.reshape(len(x1),1)
                       predicted_price= self.Linear_estimator4.predict(x1)[0]
                else:                        
                       predicted_price1= self.random_estimator3.predict(test_row)[0] 
                       x1 = test_row.full_sq
                       x1 = x1.reshape(len(x1),1)
                       predicted_price2= self.Linear_estimator3.predict(x1)[0]
                       predicted_price=.9*predicted_price1+ .1*predicted_price2
                       
        if self.specialFlag==1:
            #if (dum >70) & (dum <600) & (predicted_price > 10):
            #     predicted_price=predicted_price + 3
            if (dum > 78) :
                 predicted_price += 4
        #predicted_price -= .1
        return predicted_price 

class Kotlovka():
    #_data = pd.DataFrame
    def __init__(self, train_rows, twoclass):
        
        #### Flags
        self.specialFlag=1 ### this is for special adjustment in predict
        self.FLAG1=1  ### this is 1 for combined s2 s3
        self.FLAG2=1 ### this is 1 for global TwoClass classifier
        
        
        
        self.highcutoff = 150
        self.highcutoff1=150
        self.lowpricecutoff =4

        s1=train_rows[train_rows.full_sq > self.highcutoff];
        sa=train_rows[train_rows.full_sq < self.highcutoff];
        s2=sa[(sa.state.isnull()) & (sa.life_sq.isnull())]
        s3a = sa[~((sa.state.isnull()) & (sa.life_sq.isnull()))]
        s3=s3a[s3a.price_doc>self.lowpricecutoff]
        s4=s3a[s3a.price_doc <= self.lowpricecutoff]
        t1=s1.price_doc
        t2=s2.price_doc
        t3=s3.price_doc
        t4=s4.price_doc
        s1=s1.drop(['price_doc','sub_area','product_type'],axis=1)
        s1=s1.iloc[:,2:20]
        for fea in s1.columns:
                s1[fea].fillna(s1[fea].median(), inplace=True)
        for fea in s1.columns:
                s1[fea].fillna(0, inplace=True)
        s2=s2.drop(['price_doc','sub_area','product_type'],axis=1)
        s2=s2.iloc[:,2:20]
        for fea in s2.columns:
                s2[fea].fillna(s2[fea].median(), inplace=True)
        for fea in s2.columns:
                s2[fea].fillna(0, inplace=True)              
        s3=s3.drop(['price_doc','sub_area','product_type'],axis=1)
        s3=s3.iloc[:,2:20]
        for fea in s3.columns:
                s3[fea].fillna(s3[fea].median(), inplace=True)
        for fea in s3.columns:
                s3[fea].fillna(0, inplace=True)
        s4=s4.drop(['price_doc','sub_area','product_type'],axis=1)
        s4=s4.iloc[:,2:20]
        for fea in s4.columns:
                s4[fea].fillna(s4[fea].median(), inplace=True)
        for fea in s4.columns:
                s4[fea].fillna(0, inplace=True)
        
        sa=sa.drop(['price_doc','sub_area','product_type'],axis=1)
        sa=sa.iloc[:,2:20]
        for fea in sa.columns:
                sa[fea].fillna(sa[fea].median(), inplace=True)
        for fea in sa.columns:
                sa[fea].fillna(0, inplace=True)
                
                
        if self.FLAG1==1:
           X23 = pd.concat([s2,s3]) 
           y23 = np.concatenate([t2,t3],axis=0)
           random_estimator23=RandomForestRegressor(random_state=0, n_estimators=500)
           random_estimator23.fit(X23,y23)
           self.random_estimator2=random_estimator23
           self.random_estimator3=random_estimator23
           
           X23r=X23.full_sq
           X23r=X23r.reshape(len(X23r),1)
           y23r=y23.reshape(len(y23), 1)
           Linear_estimator23=LinearRegression()
           Linear_estimator23.fit(X23r, y23r)
           
           self.Linear_estimator2=Linear_estimator23
           self.Linear_estimator3=Linear_estimator23
           
        else:
            
            random_estimator2 = RandomForestRegressor(random_state=0, n_estimators=500)
            random_estimator2.fit(s2,t2)
            self.random_estimator2=random_estimator2
            
            Linear_estimator2 = LinearRegression()
            
            s2r=s2.full_sq
            s2r=s2r.reshape(len(s2r),1)
            t2r=t2
            t2r=t2r.reshape(len(t2r),1)
            Linear_estimator2.fit(s2r, t2r)
            self.Linear_estimator2=Linear_estimator2
            
            
            
            random_estimator3 = RandomForestRegressor(random_state=0, n_estimators=500)
            random_estimator3.fit(s3,t3)
            self.random_estimator3=random_estimator3
            
            Linear_estimator3 = LinearRegression()
            
            s3r=s3.full_sq
            s3r=s3r.reshape(len(s3r),1)
            t3r=t3
            t3r=t3r.reshape(len(t3r),1)
            Linear_estimator3.fit(s3r, t3r)
            self.Linear_estimator3=Linear_estimator3
            
            
            
           
        if self.FLAG2==1:
               self.TwoClassModel=twoclass
        else :
               
                    
                d3=np.ones(len(t3))
                d4=np.ones(len(t4))*-1
                          
                X = pd.concat([s3,s4])
                y = np.concatenate([d3,d4],axis=0)
                self.y=y
                self.X=X
                #print (X)
                #self.decision_clf = svm.SVC()
                TwoClassModel = RandomForestClassifier()
                TwoClassModel.fit(X, y)
                self.TwoClassModel=TwoClassModel
                
            
        Linear_estimator4 = LinearRegression()        
        s4r=s4.full_sq
        s4r=s4r.reshape(len(s4r),1)
        t4r=t4
        t4r=t4r.reshape(len(t4r),1)
        Linear_estimator4.fit(s4r, t4r)
        self.Linear_estimator4=Linear_estimator4             

        
    def predict_price(self,test_row):
        testrow1=test_row
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
        if dum > self.highcutoff1:
            #predicted_price=5
            predicted_price1= self.random_estimator3.predict(test_row)[0] 
            
            x1 = test_row.full_sq
            x1 = x1.reshape(len(x1),1)
            
            predicted_price2= self.Linear_estimator3.predict(x1)[0]
            
        
            predicted_price=0.2*predicted_price1+ 0.8*predicted_price2
            
            
        else:
            if flag==1:
                
                predicted_price1= self.random_estimator2.predict(test_row)[0] 
                x1 = test_row.full_sq
                x1 = x1.reshape(len(x1),1)
                predicted_price2= self.Linear_estimator2.predict(x1)[0]
                predicted_price=0.5*predicted_price1+ 0.5*predicted_price2
            
            else:
                if self.FLAG2==1:
                    testrow1=testrow1.drop(['sub_area', 'full_sq' ,'product_type'],axis=1)
                    testrow1=testrow1.iloc[:,2:20]
                    for fea in testrow1.columns:
                              testrow1[fea].fillna(testrow1[fea].median(),inplace=True)
                    for fea in testrow1.columns:
                              testrow1[fea].fillna(0,inplace=True)
                              
                    pclass=self.TwoClassModel.predict(testrow1)[0]
                
                else:
                       pclass=self.TwoClassModel.predict(test_row)[0]
                       
                if pclass==-1:
                       #predicted_price= 1.5
                       x1 = test_row.full_sq
                       x1 = x1.reshape(len(x1),1)
                       predicted_price= self.Linear_estimator4.predict(x1)[0]
                else:                        
                       predicted_price1= self.random_estimator3.predict(test_row)[0] 
                       x1 = test_row.full_sq
                       x1 = x1.reshape(len(x1),1)
                       predicted_price2= self.Linear_estimator3.predict(x1)[0]
                       predicted_price=.9*predicted_price1+ .1*predicted_price2
                       
        if self.specialFlag==1:
            #if (dum >70) & (dum <600) & (predicted_price > 10):
            #     predicted_price=predicted_price + 3
            if (dum > 78) :
                 predicted_price += 4
        #predicted_price -= .1
        return predicted_price 

gc.collect()

class Lefortovo():
    #_data = pd.DataFrame
    def __init__(self, train_rows, twoclass):
        
        #### Flags
        self.specialFlag=0 ### this is for special adjustment in predict
        self.FLAG1=1  ### this is 1 for combined s2 s3
        self.FLAG2=0 ### this is 1 for global TwoClass classifier
        
        
        
        self.highcutoff = 150
        self.highcutoff1=150
        self.lowpricecutoff =3.5

        s1=train_rows[train_rows.full_sq > self.highcutoff];
        sa=train_rows[train_rows.full_sq < self.highcutoff];
        s2=sa[(sa.state.isnull()) & (sa.life_sq.isnull())]
        s3a = sa[~((sa.state.isnull()) & (sa.life_sq.isnull()))]
        s3=s3a[s3a.price_doc>self.lowpricecutoff]
        s4=s3a[s3a.price_doc <= self.lowpricecutoff]
        t1=s1.price_doc
        t2=s2.price_doc
        t3=s3.price_doc
        t4=s4.price_doc
        s1=s1.drop(['price_doc','sub_area','product_type'],axis=1)
        s1=s1.iloc[:,2:20]
        for fea in s1.columns:
                s1[fea].fillna(s1[fea].median(), inplace=True)
        for fea in s1.columns:
                s1[fea].fillna(0, inplace=True)
        s2=s2.drop(['price_doc','sub_area','product_type'],axis=1)
        s2=s2.iloc[:,2:20]
        for fea in s2.columns:
                s2[fea].fillna(s2[fea].median(), inplace=True)
        for fea in s2.columns:
                s2[fea].fillna(0, inplace=True)              
        s3=s3.drop(['price_doc','sub_area','product_type'],axis=1)
        s3=s3.iloc[:,2:20]
        for fea in s3.columns:
                s3[fea].fillna(s3[fea].median(), inplace=True)
        for fea in s3.columns:
                s3[fea].fillna(0, inplace=True)
        s4=s4.drop(['price_doc','sub_area','product_type'],axis=1)
        s4=s4.iloc[:,2:20]
        for fea in s4.columns:
                s4[fea].fillna(s4[fea].median(), inplace=True)
        for fea in s4.columns:
                s4[fea].fillna(0, inplace=True)
        
        sa=sa.drop(['price_doc','sub_area','product_type'],axis=1)
        sa=sa.iloc[:,2:20]
        for fea in sa.columns:
                sa[fea].fillna(sa[fea].median(), inplace=True)
        for fea in sa.columns:
                sa[fea].fillna(0, inplace=True)
                
                
        if self.FLAG1==1:
           X23 = pd.concat([s2,s3]) 
           y23 = np.concatenate([t2,t3],axis=0)
           random_estimator23=RandomForestRegressor(random_state=0, n_estimators=500)
           random_estimator23.fit(X23,y23)
           self.random_estimator2=random_estimator23
           self.random_estimator3=random_estimator23
           
           X23r=X23.full_sq
           X23r=X23r.reshape(len(X23r),1)
           y23r=y23.reshape(len(y23), 1)
           Linear_estimator23=LinearRegression()
           Linear_estimator23.fit(X23r, y23r)
           
           self.Linear_estimator2=Linear_estimator23
           self.Linear_estimator3=Linear_estimator23
           
        else:
            
            random_estimator2 = RandomForestRegressor(random_state=0, n_estimators=500)
            random_estimator2.fit(s2,t2)
            self.random_estimator2=random_estimator2
            
            Linear_estimator2 = LinearRegression()
            
            s2r=s2.full_sq
            s2r=s2r.reshape(len(s2r),1)
            t2r=t2
            t2r=t2r.reshape(len(t2r),1)
            Linear_estimator2.fit(s2r, t2r)
            self.Linear_estimator2=Linear_estimator2
            
            
            
            random_estimator3 = RandomForestRegressor(random_state=0, n_estimators=500)
            random_estimator3.fit(s3,t3)
            self.random_estimator3=random_estimator3
            
            Linear_estimator3 = LinearRegression()
            
            s3r=s3.full_sq
            s3r=s3r.reshape(len(s3r),1)
            t3r=t3
            t3r=t3r.reshape(len(t3r),1)
            Linear_estimator3.fit(s3r, t3r)
            self.Linear_estimator3=Linear_estimator3
            
            
            
           
        if self.FLAG2==1:
               self.TwoClassModel=twoclass
        else :
               
                    
                d3=np.ones(len(t3))
                d4=np.ones(len(t4))*-1
                          
                X = pd.concat([s3,s4])
                y = np.concatenate([d3,d4],axis=0)
                self.y=y
                self.X=X
                #print (X)
                #self.decision_clf = svm.SVC()
                TwoClassModel = RandomForestClassifier()
                TwoClassModel.fit(X, y)
                self.TwoClassModel=TwoClassModel
                
            
        Linear_estimator4 = LinearRegression()        
        s4r=s4.full_sq
        s4r=s4r.reshape(len(s4r),1)
        t4r=t4
        t4r=t4r.reshape(len(t4r),1)
        Linear_estimator4.fit(s4r, t4r)
        self.Linear_estimator4=Linear_estimator4             

        
    def predict_price(self,test_row):
        testrow1=test_row
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
        if dum > self.highcutoff1:
            #predicted_price=5
            predicted_price1= self.random_estimator3.predict(test_row)[0] 
            
            x1 = test_row.full_sq
            x1 = x1.reshape(len(x1),1)
            
            predicted_price2= self.Linear_estimator3.predict(x1)[0]
            
        
            predicted_price=0.2*predicted_price1+ 0.8*predicted_price2
            
            
        else:
            if flag==1:
                
                predicted_price1= self.random_estimator2.predict(test_row)[0] 
                x1 = test_row.full_sq
                x1 = x1.reshape(len(x1),1)
                predicted_price2= self.Linear_estimator2.predict(x1)[0]
                predicted_price=0.5*predicted_price1+ 0.5*predicted_price2
            
            else:
                if self.FLAG2==1:
                    testrow1=testrow1.drop(['sub_area', 'full_sq' ,'product_type'],axis=1)
                    testrow1=testrow1.iloc[:,2:20]
                    for fea in testrow1.columns:
                              testrow1[fea].fillna(testrow1[fea].median(),inplace=True)
                    for fea in testrow1.columns:
                              testrow1[fea].fillna(0,inplace=True)
                              
                    pclass=self.TwoClassModel.predict(testrow1)[0]
                
                else:
                       pclass=self.TwoClassModel.predict(test_row)[0]
                       
                if pclass==-1:
                       #predicted_price= 1.5
                       x1 = test_row.full_sq
                       x1 = x1.reshape(len(x1),1)
                       predicted_price= self.Linear_estimator4.predict(x1)[0]
                else:                        
                       predicted_price1= self.random_estimator3.predict(test_row)[0] 
                       x1 = test_row.full_sq
                       x1 = x1.reshape(len(x1),1)
                       predicted_price2= self.Linear_estimator3.predict(x1)[0]
                       predicted_price=.9*predicted_price1+ .1*predicted_price2
                       
        if self.specialFlag==1:
            #if (dum >70) & (dum <600) & (predicted_price > 10):
            #     predicted_price=predicted_price + 3
            if (dum > 78) :
                 predicted_price += 4
        #predicted_price -= .1
        return predicted_price 

gc.collect()
###############################################################
class Jaroslavskoe():
    #_data = pd.DataFrame
    def __init__(self, train_rows, twoclass):
        
        #### Flags
        self.specialFlag=0 ### this is for special adjustment in predict
        self.FLAG1=1  ### this is 1 for combined s2 s3
        self.FLAG2=0 ### this is 1 for global TwoClass classifier
        
        
        
        self.highcutoff = 150
        self.highcutoff1=150
        self.lowpricecutoff =4

        s1=train_rows[train_rows.full_sq > self.highcutoff];
        sa=train_rows[train_rows.full_sq < self.highcutoff];
        s2=sa[(sa.state.isnull()) & (sa.life_sq.isnull())]
        s3a = sa[~((sa.state.isnull()) & (sa.life_sq.isnull()))]
        s3=s3a[s3a.price_doc>self.lowpricecutoff]
        s4=s3a[s3a.price_doc <= self.lowpricecutoff]
        t1=s1.price_doc
        t2=s2.price_doc
        t3=s3.price_doc
        t4=s4.price_doc
        s1=s1.drop(['price_doc','sub_area','product_type'],axis=1)
        s1=s1.iloc[:,2:20]
        for fea in s1.columns:
                s1[fea].fillna(s1[fea].median(), inplace=True)
        for fea in s1.columns:
                s1[fea].fillna(0, inplace=True)
        s2=s2.drop(['price_doc','sub_area','product_type'],axis=1)
        s2=s2.iloc[:,2:20]
        for fea in s2.columns:
                s2[fea].fillna(s2[fea].median(), inplace=True)
        for fea in s2.columns:
                s2[fea].fillna(0, inplace=True)              
        s3=s3.drop(['price_doc','sub_area','product_type'],axis=1)
        s3=s3.iloc[:,2:20]
        for fea in s3.columns:
                s3[fea].fillna(s3[fea].median(), inplace=True)
        for fea in s3.columns:
                s3[fea].fillna(0, inplace=True)
        s4=s4.drop(['price_doc','sub_area','product_type'],axis=1)
        s4=s4.iloc[:,2:20]
        for fea in s4.columns:
                s4[fea].fillna(s4[fea].median(), inplace=True)
        for fea in s4.columns:
                s4[fea].fillna(0, inplace=True)
        
        sa=sa.drop(['price_doc','sub_area','product_type'],axis=1)
        sa=sa.iloc[:,2:20]
        for fea in sa.columns:
                sa[fea].fillna(sa[fea].median(), inplace=True)
        for fea in sa.columns:
                sa[fea].fillna(0, inplace=True)
                
                
        if self.FLAG1==1:
           X23 = pd.concat([s2,s3]) 
           y23 = np.concatenate([t2,t3],axis=0)
           random_estimator23=RandomForestRegressor(random_state=0, n_estimators=500)
           random_estimator23.fit(X23,y23)
           self.random_estimator2=random_estimator23
           self.random_estimator3=random_estimator23
           
           X23r=X23.full_sq
           X23r=X23r.reshape(len(X23r),1)
           y23r=y23.reshape(len(y23), 1)
           Linear_estimator23=LinearRegression()
           Linear_estimator23.fit(X23r, y23r)
           
           self.Linear_estimator2=Linear_estimator23
           self.Linear_estimator3=Linear_estimator23
           
        else:
            
            random_estimator2 = RandomForestRegressor(random_state=0, n_estimators=500)
            random_estimator2.fit(s2,t2)
            self.random_estimator2=random_estimator2
            
            Linear_estimator2 = LinearRegression()
            
            s2r=s2.full_sq
            s2r=s2r.reshape(len(s2r),1)
            t2r=t2
            t2r=t2r.reshape(len(t2r),1)
            Linear_estimator2.fit(s2r, t2r)
            self.Linear_estimator2=Linear_estimator2
            
            
            
            random_estimator3 = RandomForestRegressor(random_state=0, n_estimators=500)
            random_estimator3.fit(s3,t3)
            self.random_estimator3=random_estimator3
            
            Linear_estimator3 = LinearRegression()
            
            s3r=s3.full_sq
            s3r=s3r.reshape(len(s3r),1)
            t3r=t3
            t3r=t3r.reshape(len(t3r),1)
            Linear_estimator3.fit(s3r, t3r)
            self.Linear_estimator3=Linear_estimator3
            
            
            
           
        if self.FLAG2==1:
               self.TwoClassModel=twoclass
        else :
               
                    
                d3=np.ones(len(t3))
                d4=np.ones(len(t4))*-1
                          
                X = pd.concat([s3,s4])
                y = np.concatenate([d3,d4],axis=0)
                self.y=y
                self.X=X
                #print (X)
                #self.decision_clf = svm.SVC()
                TwoClassModel = RandomForestClassifier()
                TwoClassModel.fit(X, y)
                self.TwoClassModel=TwoClassModel
                
            
        Linear_estimator4 = LinearRegression()        
        s4r=s4.full_sq
        s4r=s4r.reshape(len(s4r),1)
        t4r=t4
        t4r=t4r.reshape(len(t4r),1)
        Linear_estimator4.fit(s4r, t4r)
        self.Linear_estimator4=Linear_estimator4             

        
    def predict_price(self,test_row):
        testrow1=test_row
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
        if dum > self.highcutoff1:
            #predicted_price=5
            predicted_price1= self.random_estimator3.predict(test_row)[0] 
            
            x1 = test_row.full_sq
            x1 = x1.reshape(len(x1),1)
            
            predicted_price2= self.Linear_estimator3.predict(x1)[0]
            
        
            predicted_price=0.2*predicted_price1+ 0.8*predicted_price2
            
            
        else:
            if flag==1:
                
                predicted_price1= self.random_estimator2.predict(test_row)[0] 
                x1 = test_row.full_sq
                x1 = x1.reshape(len(x1),1)
                predicted_price2= self.Linear_estimator2.predict(x1)[0]
                predicted_price=0.5*predicted_price1+ 0.5*predicted_price2
            
            else:
                if self.FLAG2==1:
                    testrow1=testrow1.drop(['sub_area', 'full_sq' ,'product_type'],axis=1)
                    testrow1=testrow1.iloc[:,2:20]
                    for fea in testrow1.columns:
                              testrow1[fea].fillna(testrow1[fea].median(),inplace=True)
                    for fea in testrow1.columns:
                              testrow1[fea].fillna(0,inplace=True)
                              
                    pclass=self.TwoClassModel.predict(testrow1)[0]
                
                else:
                       pclass=self.TwoClassModel.predict(test_row)[0]
                       
                if pclass==-1:
                       #predicted_price= 1.5
                       x1 = test_row.full_sq
                       x1 = x1.reshape(len(x1),1)
                       predicted_price= self.Linear_estimator4.predict(x1)[0]
                else:                        
                       predicted_price1= self.random_estimator3.predict(test_row)[0] 
                       x1 = test_row.full_sq
                       x1 = x1.reshape(len(x1),1)
                       predicted_price2= self.Linear_estimator3.predict(x1)[0]
                       predicted_price=.9*predicted_price1+ .1*predicted_price2
                       
        if self.specialFlag==1:
            #if (dum >70) & (dum <600) & (predicted_price > 10):
            #     predicted_price=predicted_price + 3
            if (dum > 78) :
                 predicted_price += 4
        #predicted_price -= .1
        return predicted_price 

gc.collect()
###############################################################
class Ochakovo_Matveevskoe():
    #_data = pd.DataFrame
    def __init__(self, train_rows, twoclass):
        
        #### Flags
        self.specialFlag=0 ### this is for special adjustment in predict
        self.FLAG1=1  ### this is 1 for combined s2 s3
        self.FLAG2=0 ### this is 1 for global TwoClass classifier
        
        
        
        self.highcutoff = 150
        self.highcutoff1=150
        self.lowpricecutoff =4.5

        s1=train_rows[train_rows.full_sq > self.highcutoff];
        sa=train_rows[train_rows.full_sq < self.highcutoff];
        s2=sa[(sa.state.isnull()) & (sa.life_sq.isnull())]
        s3a = sa[~((sa.state.isnull()) & (sa.life_sq.isnull()))]
        s3=s3a[s3a.price_doc>self.lowpricecutoff]
        s4=s3a[s3a.price_doc <= self.lowpricecutoff]
        t1=s1.price_doc
        t2=s2.price_doc
        t3=s3.price_doc
        t4=s4.price_doc
        s1=s1.drop(['price_doc','sub_area','product_type'],axis=1)
        s1=s1.iloc[:,2:20]
        for fea in s1.columns:
                s1[fea].fillna(s1[fea].median(), inplace=True)
        for fea in s1.columns:
                s1[fea].fillna(0, inplace=True)
        s2=s2.drop(['price_doc','sub_area','product_type'],axis=1)
        s2=s2.iloc[:,2:20]
        for fea in s2.columns:
                s2[fea].fillna(s2[fea].median(), inplace=True)
        for fea in s2.columns:
                s2[fea].fillna(0, inplace=True)              
        s3=s3.drop(['price_doc','sub_area','product_type'],axis=1)
        s3=s3.iloc[:,2:20]
        for fea in s3.columns:
                s3[fea].fillna(s3[fea].median(), inplace=True)
        for fea in s3.columns:
                s3[fea].fillna(0, inplace=True)
        s4=s4.drop(['price_doc','sub_area','product_type'],axis=1)
        s4=s4.iloc[:,2:20]
        for fea in s4.columns:
                s4[fea].fillna(s4[fea].median(), inplace=True)
        for fea in s4.columns:
                s4[fea].fillna(0, inplace=True)
        
        sa=sa.drop(['price_doc','sub_area','product_type'],axis=1)
        sa=sa.iloc[:,2:20]
        for fea in sa.columns:
                sa[fea].fillna(sa[fea].median(), inplace=True)
        for fea in sa.columns:
                sa[fea].fillna(0, inplace=True)
                
                
        if self.FLAG1==1:
           X23 = pd.concat([s2,s3]) 
           y23 = np.concatenate([t2,t3],axis=0)
           random_estimator23=RandomForestRegressor(random_state=0, n_estimators=500)
           random_estimator23.fit(X23,y23)
           self.random_estimator2=random_estimator23
           self.random_estimator3=random_estimator23
           
           X23r=X23.full_sq
           X23r=X23r.reshape(len(X23r),1)
           y23r=y23.reshape(len(y23), 1)
           Linear_estimator23=LinearRegression()
           Linear_estimator23.fit(X23r, y23r)
           
           self.Linear_estimator2=Linear_estimator23
           self.Linear_estimator3=Linear_estimator23
           
        else:
            
            random_estimator2 = RandomForestRegressor(random_state=0, n_estimators=500)
            random_estimator2.fit(s2,t2)
            self.random_estimator2=random_estimator2
            
            Linear_estimator2 = LinearRegression()
            
            s2r=s2.full_sq
            s2r=s2r.reshape(len(s2r),1)
            t2r=t2
            t2r=t2r.reshape(len(t2r),1)
            Linear_estimator2.fit(s2r, t2r)
            self.Linear_estimator2=Linear_estimator2
            
            
            
            random_estimator3 = RandomForestRegressor(random_state=0, n_estimators=500)
            random_estimator3.fit(s3,t3)
            self.random_estimator3=random_estimator3
            
            Linear_estimator3 = LinearRegression()
            
            s3r=s3.full_sq
            s3r=s3r.reshape(len(s3r),1)
            t3r=t3
            t3r=t3r.reshape(len(t3r),1)
            Linear_estimator3.fit(s3r, t3r)
            self.Linear_estimator3=Linear_estimator3
            
            
            
           
        if self.FLAG2==1:
               self.TwoClassModel=twoclass
        else :
               
                    
                d3=np.ones(len(t3))
                d4=np.ones(len(t4))*-1
                          
                X = pd.concat([s3,s4])
                y = np.concatenate([d3,d4],axis=0)
                self.y=y
                self.X=X
                #print (X)
                #self.decision_clf = svm.SVC()
                TwoClassModel = RandomForestClassifier()
                TwoClassModel.fit(X, y)
                self.TwoClassModel=TwoClassModel
                
            
        Linear_estimator4 = LinearRegression()        
        s4r=s4.full_sq
        s4r=s4r.reshape(len(s4r),1)
        t4r=t4
        t4r=t4r.reshape(len(t4r),1)
        Linear_estimator4.fit(s4r, t4r)
        self.Linear_estimator4=Linear_estimator4             

        
    def predict_price(self,test_row):
        testrow1=test_row
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
        if dum > self.highcutoff1:
            #predicted_price=5
            predicted_price1= self.random_estimator3.predict(test_row)[0] 
            
            x1 = test_row.full_sq
            x1 = x1.reshape(len(x1),1)
            
            predicted_price2= self.Linear_estimator3.predict(x1)[0]
            
        
            predicted_price=0.2*predicted_price1+ 0.8*predicted_price2
            
            
        else:
            if flag==1:
                
                predicted_price1= self.random_estimator2.predict(test_row)[0] 
                x1 = test_row.full_sq
                x1 = x1.reshape(len(x1),1)
                predicted_price2= self.Linear_estimator2.predict(x1)[0]
                predicted_price=0.5*predicted_price1+ 0.5*predicted_price2
            
            else:
                if self.FLAG2==1:
                    testrow1=testrow1.drop(['sub_area', 'full_sq' ,'product_type'],axis=1)
                    testrow1=testrow1.iloc[:,2:20]
                    for fea in testrow1.columns:
                              testrow1[fea].fillna(testrow1[fea].median(),inplace=True)
                    for fea in testrow1.columns:
                              testrow1[fea].fillna(0,inplace=True)
                              
                    pclass=self.TwoClassModel.predict(testrow1)[0]
                
                else:
                       pclass=self.TwoClassModel.predict(test_row)[0]
                       
                if pclass==-1:
                       #predicted_price= 1.5
                       x1 = test_row.full_sq
                       x1 = x1.reshape(len(x1),1)
                       predicted_price= self.Linear_estimator4.predict(x1)[0]
                else:                        
                       predicted_price1= self.random_estimator3.predict(test_row)[0] 
                       x1 = test_row.full_sq
                       x1 = x1.reshape(len(x1),1)
                       predicted_price2= self.Linear_estimator3.predict(x1)[0]
                       predicted_price=.9*predicted_price1+ .1*predicted_price2
                       
        if self.specialFlag==1:
            #if (dum >70) & (dum <600) & (predicted_price > 10):
            #     predicted_price=predicted_price + 3
            if (dum > 78) :
                 predicted_price += 4
        #predicted_price -= .1
        return predicted_price 

gc.collect()
###############################################################

class Perovo():
    #_data = pd.DataFrame
    def __init__(self, train_rows, twoclass):
        
        #### Flags
        self.specialFlag=1 ### this is for special adjustment in predict
        self.FLAG1=1  ### this is 1 for combined s2 s3
        self.FLAG2=0 ### this is 1 for global TwoClass classifier
        
        
        
        self.highcutoff = 150
        self.highcutoff1=150
        self.lowpricecutoff =3.2

        s1=train_rows[train_rows.full_sq > self.highcutoff];
        sa=train_rows[train_rows.full_sq < self.highcutoff];
        s2=sa[(sa.state.isnull()) & (sa.life_sq.isnull())]
        s3a = sa[~((sa.state.isnull()) & (sa.life_sq.isnull()))]
        s3=s3a[s3a.price_doc>self.lowpricecutoff]
        s4=s3a[s3a.price_doc <= self.lowpricecutoff]
        t1=s1.price_doc
        t2=s2.price_doc
        t3=s3.price_doc
        t4=s4.price_doc
        s1=s1.drop(['price_doc','sub_area','product_type'],axis=1)
        s1=s1.iloc[:,2:20]
        for fea in s1.columns:
                s1[fea].fillna(s1[fea].median(), inplace=True)
        for fea in s1.columns:
                s1[fea].fillna(0, inplace=True)
        s2=s2.drop(['price_doc','sub_area','product_type'],axis=1)
        s2=s2.iloc[:,2:20]
        for fea in s2.columns:
                s2[fea].fillna(s2[fea].median(), inplace=True)
        for fea in s2.columns:
                s2[fea].fillna(0, inplace=True)              
        s3=s3.drop(['price_doc','sub_area','product_type'],axis=1)
        s3=s3.iloc[:,2:20]
        for fea in s3.columns:
                s3[fea].fillna(s3[fea].median(), inplace=True)
        for fea in s3.columns:
                s3[fea].fillna(0, inplace=True)
        s4=s4.drop(['price_doc','sub_area','product_type'],axis=1)
        s4=s4.iloc[:,2:20]
        for fea in s4.columns:
                s4[fea].fillna(s4[fea].median(), inplace=True)
        for fea in s4.columns:
                s4[fea].fillna(0, inplace=True)
        
        sa=sa.drop(['price_doc','sub_area','product_type'],axis=1)
        sa=sa.iloc[:,2:20]
        for fea in sa.columns:
                sa[fea].fillna(sa[fea].median(), inplace=True)
        for fea in sa.columns:
                sa[fea].fillna(0, inplace=True)
                
                
        if self.FLAG1==1:
           X23 = pd.concat([s2,s3]) 
           y23 = np.concatenate([t2,t3],axis=0)
           random_estimator23=RandomForestRegressor(random_state=0, n_estimators=500)
           random_estimator23.fit(X23,y23)
           self.random_estimator2=random_estimator23
           self.random_estimator3=random_estimator23
           
           X23r=X23.full_sq
           X23r=X23r.reshape(len(X23r),1)
           y23r=y23.reshape(len(y23), 1)
           Linear_estimator23=LinearRegression()
           Linear_estimator23.fit(X23r, y23r)
           
           self.Linear_estimator2=Linear_estimator23
           self.Linear_estimator3=Linear_estimator23
           
        else:
            
            random_estimator2 = RandomForestRegressor(random_state=0, n_estimators=500)
            random_estimator2.fit(s2,t2)
            self.random_estimator2=random_estimator2
            
            Linear_estimator2 = LinearRegression()
            
            s2r=s2.full_sq
            s2r=s2r.reshape(len(s2r),1)
            t2r=t2
            t2r=t2r.reshape(len(t2r),1)
            Linear_estimator2.fit(s2r, t2r)
            self.Linear_estimator2=Linear_estimator2
            
            
            
            random_estimator3 = RandomForestRegressor(random_state=0, n_estimators=500)
            random_estimator3.fit(s3,t3)
            self.random_estimator3=random_estimator3
            
            Linear_estimator3 = LinearRegression()
            
            s3r=s3.full_sq
            s3r=s3r.reshape(len(s3r),1)
            t3r=t3
            t3r=t3r.reshape(len(t3r),1)
            Linear_estimator3.fit(s3r, t3r)
            self.Linear_estimator3=Linear_estimator3
            
            
            
           
        if self.FLAG2==1:
               self.TwoClassModel=twoclass
        else :
               
                    
                d3=np.ones(len(t3))
                d4=np.ones(len(t4))*-1
                          
                X = pd.concat([s3,s4])
                y = np.concatenate([d3,d4],axis=0)
                self.y=y
                self.X=X
                #print (X)
                #self.decision_clf = svm.SVC()
                TwoClassModel = RandomForestClassifier()
                TwoClassModel.fit(X, y)
                self.TwoClassModel=TwoClassModel
                
            
        Linear_estimator4 = LinearRegression()        
        s4r=s4.full_sq
        s4r=s4r.reshape(len(s4r),1)
        t4r=t4
        t4r=t4r.reshape(len(t4r),1)
        Linear_estimator4.fit(s4r, t4r)
        self.Linear_estimator4=Linear_estimator4             

        
    def predict_price(self,test_row):
        testrow1=test_row
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
        if dum > self.highcutoff1:
            #predicted_price=5
            predicted_price1= self.random_estimator3.predict(test_row)[0] 
            
            x1 = test_row.full_sq
            x1 = x1.reshape(len(x1),1)
            
            predicted_price2= self.Linear_estimator3.predict(x1)[0]
            
        
            predicted_price=0.2*predicted_price1+ 0.8*predicted_price2
            
            
        else:
            if flag==1:
                
                predicted_price1= self.random_estimator2.predict(test_row)[0] 
                x1 = test_row.full_sq
                x1 = x1.reshape(len(x1),1)
                predicted_price2= self.Linear_estimator2.predict(x1)[0]
                predicted_price=0.5*predicted_price1+ 0.5*predicted_price2
            
            else:
                if self.FLAG2==1:
                    testrow1=testrow1.drop(['sub_area', 'full_sq' ,'product_type'],axis=1)
                    testrow1=testrow1.iloc[:,2:20]
                    for fea in testrow1.columns:
                              testrow1[fea].fillna(testrow1[fea].median(),inplace=True)
                    for fea in testrow1.columns:
                              testrow1[fea].fillna(0,inplace=True)
                              
                    pclass=self.TwoClassModel.predict(testrow1)[0]
                
                else:
                       pclass=self.TwoClassModel.predict(test_row)[0]
                       
                if pclass==-1:
                       #predicted_price= 1.5
                       x1 = test_row.full_sq
                       x1 = x1.reshape(len(x1),1)
                       predicted_price= self.Linear_estimator4.predict(x1)[0]
                else:                        
                       predicted_price1= self.random_estimator3.predict(test_row)[0] 
                       x1 = test_row.full_sq
                       x1 = x1.reshape(len(x1),1)
                       predicted_price2= self.Linear_estimator3.predict(x1)[0]
                       predicted_price=.9*predicted_price1+ .1*predicted_price2
                       
        if self.specialFlag==1:
            #if (dum >70) & (dum <600) & (predicted_price > 10):
            #     predicted_price=predicted_price + 3
            if (dum > 80) & (predicted_price > 8) :
                 predicted_price += 4
        #predicted_price -= .1
        return predicted_price 

gc.collect()
###############################################################
class Nizhegorodskoe():
    #_data = pd.DataFrame
    def __init__(self, train_rows, twoclass):
        
        #### Flags
        self.specialFlag=0 ### this is for special adjustment in predict
        self.FLAG1=1  ### this is 1 for combined s2 s3
        self.FLAG2=0 ### this is 1 for global TwoClass classifier
        
        
        
        self.highcutoff = 150
        self.highcutoff1=150
        self.lowpricecutoff =2.6

        s1=train_rows[train_rows.full_sq > self.highcutoff];
        sa=train_rows[train_rows.full_sq < self.highcutoff];
        s2=sa[(sa.state.isnull()) & (sa.life_sq.isnull())]
        s3a = sa[~((sa.state.isnull()) & (sa.life_sq.isnull()))]
        s3=s3a[s3a.price_doc>self.lowpricecutoff]
        s4=s3a[s3a.price_doc <= self.lowpricecutoff]
        t1=s1.price_doc
        t2=s2.price_doc
        t3=s3.price_doc
        t4=s4.price_doc
        s1=s1.drop(['price_doc','sub_area','product_type'],axis=1)
        s1=s1.iloc[:,2:20]
        for fea in s1.columns:
                s1[fea].fillna(s1[fea].median(), inplace=True)
        for fea in s1.columns:
                s1[fea].fillna(0, inplace=True)
        s2=s2.drop(['price_doc','sub_area','product_type'],axis=1)
        s2=s2.iloc[:,2:20]
        for fea in s2.columns:
                s2[fea].fillna(s2[fea].median(), inplace=True)
        for fea in s2.columns:
                s2[fea].fillna(0, inplace=True)              
        s3=s3.drop(['price_doc','sub_area','product_type'],axis=1)
        s3=s3.iloc[:,2:20]
        for fea in s3.columns:
                s3[fea].fillna(s3[fea].median(), inplace=True)
        for fea in s3.columns:
                s3[fea].fillna(0, inplace=True)
        s4=s4.drop(['price_doc','sub_area','product_type'],axis=1)
        s4=s4.iloc[:,2:20]
        for fea in s4.columns:
                s4[fea].fillna(s4[fea].median(), inplace=True)
        for fea in s4.columns:
                s4[fea].fillna(0, inplace=True)
        
        sa=sa.drop(['price_doc','sub_area','product_type'],axis=1)
        sa=sa.iloc[:,2:20]
        for fea in sa.columns:
                sa[fea].fillna(sa[fea].median(), inplace=True)
        for fea in sa.columns:
                sa[fea].fillna(0, inplace=True)
                
                
        if self.FLAG1==1:
           X23 = pd.concat([s2,s3]) 
           y23 = np.concatenate([t2,t3],axis=0)
           random_estimator23=RandomForestRegressor(random_state=0, n_estimators=500)
           random_estimator23.fit(X23,y23)
           self.random_estimator2=random_estimator23
           self.random_estimator3=random_estimator23
           
           X23r=X23.full_sq
           X23r=X23r.reshape(len(X23r),1)
           y23r=y23.reshape(len(y23), 1)
           Linear_estimator23=LinearRegression()
           Linear_estimator23.fit(X23r, y23r)
           
           self.Linear_estimator2=Linear_estimator23
           self.Linear_estimator3=Linear_estimator23
           
        else:
            
            random_estimator2 = RandomForestRegressor(random_state=0, n_estimators=500)
            random_estimator2.fit(s2,t2)
            self.random_estimator2=random_estimator2
            
            Linear_estimator2 = LinearRegression()
            
            s2r=s2.full_sq
            s2r=s2r.reshape(len(s2r),1)
            t2r=t2
            t2r=t2r.reshape(len(t2r),1)
            Linear_estimator2.fit(s2r, t2r)
            self.Linear_estimator2=Linear_estimator2
            
            
            
            random_estimator3 = RandomForestRegressor(random_state=0, n_estimators=500)
            random_estimator3.fit(s3,t3)
            self.random_estimator3=random_estimator3
            
            Linear_estimator3 = LinearRegression()
            
            s3r=s3.full_sq
            s3r=s3r.reshape(len(s3r),1)
            t3r=t3
            t3r=t3r.reshape(len(t3r),1)
            Linear_estimator3.fit(s3r, t3r)
            self.Linear_estimator3=Linear_estimator3
            
            
            
           
        if self.FLAG2==1:
               self.TwoClassModel=twoclass
        else :
               
                    
                d3=np.ones(len(t3))
                d4=np.ones(len(t4))*-1
                          
                X = pd.concat([s3,s4])
                y = np.concatenate([d3,d4],axis=0)
                self.y=y
                self.X=X
                #print (X)
                #self.decision_clf = svm.SVC()
                TwoClassModel = RandomForestClassifier()
                TwoClassModel.fit(X, y)
                self.TwoClassModel=TwoClassModel
                
            
        Linear_estimator4 = LinearRegression()        
        s4r=s4.full_sq
        s4r=s4r.reshape(len(s4r),1)
        t4r=t4
        t4r=t4r.reshape(len(t4r),1)
        Linear_estimator4.fit(s4r, t4r)
        self.Linear_estimator4=Linear_estimator4             

        
    def predict_price(self,test_row):
        testrow1=test_row
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
        if dum > self.highcutoff1:
            #predicted_price=5
            predicted_price1= self.random_estimator3.predict(test_row)[0] 
            
            x1 = test_row.full_sq
            x1 = x1.reshape(len(x1),1)
            
            predicted_price2= self.Linear_estimator3.predict(x1)[0]
            
        
            predicted_price=0.2*predicted_price1+ 0.8*predicted_price2
            
            
        else:
            if flag==1:
                
                predicted_price1= self.random_estimator2.predict(test_row)[0] 
                x1 = test_row.full_sq
                x1 = x1.reshape(len(x1),1)
                predicted_price2= self.Linear_estimator2.predict(x1)[0]
                predicted_price=0.5*predicted_price1+ 0.5*predicted_price2
            
            else:
                if self.FLAG2==1:
                    testrow1=testrow1.drop(['sub_area', 'full_sq' ,'product_type'],axis=1)
                    testrow1=testrow1.iloc[:,2:20]
                    for fea in testrow1.columns:
                              testrow1[fea].fillna(testrow1[fea].median(),inplace=True)
                    for fea in testrow1.columns:
                              testrow1[fea].fillna(0,inplace=True)
                              
                    pclass=self.TwoClassModel.predict(testrow1)[0]
                
                else:
                       pclass=self.TwoClassModel.predict(test_row)[0]
                       
                if pclass==-1:
                       #predicted_price= 1.5
                       x1 = test_row.full_sq
                       x1 = x1.reshape(len(x1),1)
                       predicted_price= self.Linear_estimator4.predict(x1)[0]
                else:                        
                       predicted_price1= self.random_estimator3.predict(test_row)[0] 
                       x1 = test_row.full_sq
                       x1 = x1.reshape(len(x1),1)
                       predicted_price2= self.Linear_estimator3.predict(x1)[0]
                       predicted_price=.9*predicted_price1+ .1*predicted_price2
                       
        if self.specialFlag==1:
            #if (dum >70) & (dum <600) & (predicted_price > 10):
            #     predicted_price=predicted_price + 3
            if (dum > 80) & (predicted_price > 8) :
                 predicted_price += 4
        #predicted_price -= .1
        return predicted_price 

gc.collect()
###############################################################
class Ivanovskoe():
    #_data = pd.DataFrame
    def __init__(self, train_rows, twoclass):
        
        #### Flags
        self.specialFlag=1 ### this is for special adjustment in predict
        self.FLAG1=1  ### this is 1 for combined s2 s3
        self.FLAG2=0 ### this is 1 for global TwoClass classifier
        
        
        
        self.highcutoff = 150
        self.highcutoff1=150
        self.lowpricecutoff =3.7

        s1=train_rows[train_rows.full_sq > self.highcutoff];
        sa=train_rows[train_rows.full_sq < self.highcutoff];
        s2=sa[(sa.state.isnull()) & (sa.life_sq.isnull())]
        s3a = sa[~((sa.state.isnull()) & (sa.life_sq.isnull()))]
        s3=s3a[s3a.price_doc>self.lowpricecutoff]
        s4=s3a[s3a.price_doc <= self.lowpricecutoff]
        t1=s1.price_doc
        t2=s2.price_doc
        t3=s3.price_doc
        t4=s4.price_doc
        s1=s1.drop(['price_doc','sub_area','product_type'],axis=1)
        s1=s1.iloc[:,2:20]
        for fea in s1.columns:
                s1[fea].fillna(s1[fea].median(), inplace=True)
        for fea in s1.columns:
                s1[fea].fillna(0, inplace=True)
        s2=s2.drop(['price_doc','sub_area','product_type'],axis=1)
        s2=s2.iloc[:,2:20]
        for fea in s2.columns:
                s2[fea].fillna(s2[fea].median(), inplace=True)
        for fea in s2.columns:
                s2[fea].fillna(0, inplace=True)              
        s3=s3.drop(['price_doc','sub_area','product_type'],axis=1)
        s3=s3.iloc[:,2:20]
        for fea in s3.columns:
                s3[fea].fillna(s3[fea].median(), inplace=True)
        for fea in s3.columns:
                s3[fea].fillna(0, inplace=True)
        s4=s4.drop(['price_doc','sub_area','product_type'],axis=1)
        s4=s4.iloc[:,2:20]
        for fea in s4.columns:
                s4[fea].fillna(s4[fea].median(), inplace=True)
        for fea in s4.columns:
                s4[fea].fillna(0, inplace=True)
        
        sa=sa.drop(['price_doc','sub_area','product_type'],axis=1)
        sa=sa.iloc[:,2:20]
        for fea in sa.columns:
                sa[fea].fillna(sa[fea].median(), inplace=True)
        for fea in sa.columns:
                sa[fea].fillna(0, inplace=True)
                
                
        if self.FLAG1==1:
           X23 = pd.concat([s2,s3]) 
           y23 = np.concatenate([t2,t3],axis=0)
           random_estimator23=RandomForestRegressor(random_state=0, n_estimators=500)
           random_estimator23.fit(X23,y23)
           self.random_estimator2=random_estimator23
           self.random_estimator3=random_estimator23
           
           X23r=X23.full_sq
           X23r=X23r.reshape(len(X23r),1)
           y23r=y23.reshape(len(y23), 1)
           Linear_estimator23=LinearRegression()
           Linear_estimator23.fit(X23r, y23r)
           
           self.Linear_estimator2=Linear_estimator23
           self.Linear_estimator3=Linear_estimator23
           
        else:
            
            random_estimator2 = RandomForestRegressor(random_state=0, n_estimators=500)
            random_estimator2.fit(s2,t2)
            self.random_estimator2=random_estimator2
            
            Linear_estimator2 = LinearRegression()
            
            s2r=s2.full_sq
            s2r=s2r.reshape(len(s2r),1)
            t2r=t2
            t2r=t2r.reshape(len(t2r),1)
            Linear_estimator2.fit(s2r, t2r)
            self.Linear_estimator2=Linear_estimator2
            
            
            
            random_estimator3 = RandomForestRegressor(random_state=0, n_estimators=500)
            random_estimator3.fit(s3,t3)
            self.random_estimator3=random_estimator3
            
            Linear_estimator3 = LinearRegression()
            
            s3r=s3.full_sq
            s3r=s3r.reshape(len(s3r),1)
            t3r=t3
            t3r=t3r.reshape(len(t3r),1)
            Linear_estimator3.fit(s3r, t3r)
            self.Linear_estimator3=Linear_estimator3
            
            
            
           
        if self.FLAG2==1:
               self.TwoClassModel=twoclass
        else :
               
                    
                d3=np.ones(len(t3))
                d4=np.ones(len(t4))*-1
                          
                X = pd.concat([s3,s4])
                y = np.concatenate([d3,d4],axis=0)
                self.y=y
                self.X=X
                #print (X)
                #self.decision_clf = svm.SVC()
                TwoClassModel = RandomForestClassifier()
                TwoClassModel.fit(X, y)
                self.TwoClassModel=TwoClassModel
                
            
        Linear_estimator4 = LinearRegression()        
        s4r=s4.full_sq
        s4r=s4r.reshape(len(s4r),1)
        t4r=t4
        t4r=t4r.reshape(len(t4r),1)
        Linear_estimator4.fit(s4r, t4r)
        self.Linear_estimator4=Linear_estimator4             

        
    def predict_price(self,test_row):
        testrow1=test_row
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
        if dum > self.highcutoff1:
            #predicted_price=5
            predicted_price1= self.random_estimator3.predict(test_row)[0] 
            
            x1 = test_row.full_sq
            x1 = x1.reshape(len(x1),1)
            
            predicted_price2= self.Linear_estimator3.predict(x1)[0]
            
        
            predicted_price=0.2*predicted_price1+ 0.8*predicted_price2
            
            
        else:
            if flag==1:
                
                predicted_price1= self.random_estimator2.predict(test_row)[0] 
                x1 = test_row.full_sq
                x1 = x1.reshape(len(x1),1)
                predicted_price2= self.Linear_estimator2.predict(x1)[0]
                predicted_price=0.5*predicted_price1+ 0.5*predicted_price2
            
            else:
                if self.FLAG2==1:
                    testrow1=testrow1.drop(['sub_area', 'full_sq' ,'product_type'],axis=1)
                    testrow1=testrow1.iloc[:,2:20]
                    for fea in testrow1.columns:
                              testrow1[fea].fillna(testrow1[fea].median(),inplace=True)
                    for fea in testrow1.columns:
                              testrow1[fea].fillna(0,inplace=True)
                              
                    pclass=self.TwoClassModel.predict(testrow1)[0]
                
                else:
                       pclass=self.TwoClassModel.predict(test_row)[0]
                       
                if pclass==-1:
                       #predicted_price= 1.5
                       x1 = test_row.full_sq
                       x1 = x1.reshape(len(x1),1)
                       predicted_price= self.Linear_estimator4.predict(x1)[0]
                else:                        
                       predicted_price1= self.random_estimator3.predict(test_row)[0] 
                       x1 = test_row.full_sq
                       x1 = x1.reshape(len(x1),1)
                       predicted_price2= self.Linear_estimator3.predict(x1)[0]
                       predicted_price=.9*predicted_price1+ .1*predicted_price2
                       
        if self.specialFlag==1:
            #if (dum >70) & (dum <600) & (predicted_price > 10):
            #     predicted_price=predicted_price + 3
            if (dum > 120)  :
                 predicted_price = 16
        #predicted_price -= .1
        return predicted_price 

gc.collect()
###############################################################
class Severnoe_Tushino():
    #_data = pd.DataFrame
    def __init__(self, train_rows, twoclass):
        
        #### Flags
        self.specialFlag=1 ### this is for special adjustment in predict
        self.FLAG1=0  ### this is 1 for combined s2 s3
        self.FLAG2=0 ### this is 1 for global TwoClass classifier
        
        
        
        self.highcutoff = 150
        self.highcutoff1=150
        self.lowpricecutoff =3.7

        s1=train_rows[train_rows.full_sq > self.highcutoff];
        sa=train_rows[train_rows.full_sq < self.highcutoff];
        s2=sa[(sa.state.isnull()) & (sa.life_sq.isnull())]
        s3a = sa[~((sa.state.isnull()) & (sa.life_sq.isnull()))]
        s3=s3a[s3a.price_doc>self.lowpricecutoff]
        s4=s3a[s3a.price_doc <= self.lowpricecutoff]
        t1=s1.price_doc
        t2=s2.price_doc
        t3=s3.price_doc
        t4=s4.price_doc
        s1=s1.drop(['price_doc','sub_area','product_type'],axis=1)
        s1=s1.iloc[:,2:20]
        for fea in s1.columns:
                s1[fea].fillna(s1[fea].median(), inplace=True)
        for fea in s1.columns:
                s1[fea].fillna(0, inplace=True)
        s2=s2.drop(['price_doc','sub_area','product_type'],axis=1)
        s2=s2.iloc[:,2:20]
        for fea in s2.columns:
                s2[fea].fillna(s2[fea].median(), inplace=True)
        for fea in s2.columns:
                s2[fea].fillna(0, inplace=True)              
        s3=s3.drop(['price_doc','sub_area','product_type'],axis=1)
        s3=s3.iloc[:,2:20]
        for fea in s3.columns:
                s3[fea].fillna(s3[fea].median(), inplace=True)
        for fea in s3.columns:
                s3[fea].fillna(0, inplace=True)
        s4=s4.drop(['price_doc','sub_area','product_type'],axis=1)
        s4=s4.iloc[:,2:20]
        for fea in s4.columns:
                s4[fea].fillna(s4[fea].median(), inplace=True)
        for fea in s4.columns:
                s4[fea].fillna(0, inplace=True)
        
        sa=sa.drop(['price_doc','sub_area','product_type'],axis=1)
        sa=sa.iloc[:,2:20]
        for fea in sa.columns:
                sa[fea].fillna(sa[fea].median(), inplace=True)
        for fea in sa.columns:
                sa[fea].fillna(0, inplace=True)
                
                
        if self.FLAG1==1:
           X23 = pd.concat([s2,s3]) 
           y23 = np.concatenate([t2,t3],axis=0)
           random_estimator23=RandomForestRegressor(random_state=0, n_estimators=500)
           random_estimator23.fit(X23,y23)
           self.random_estimator2=random_estimator23
           self.random_estimator3=random_estimator23
           
           X23r=X23.full_sq
           X23r=X23r.reshape(len(X23r),1)
           y23r=y23.reshape(len(y23), 1)
           Linear_estimator23=LinearRegression()
           Linear_estimator23.fit(X23r, y23r)
           
           self.Linear_estimator2=Linear_estimator23
           self.Linear_estimator3=Linear_estimator23
           
        else:
            
            random_estimator2 = RandomForestRegressor(random_state=0, n_estimators=500)
            random_estimator2.fit(s2,t2)
            self.random_estimator2=random_estimator2
            
            Linear_estimator2 = LinearRegression()
            
            s2r=s2.full_sq
            s2r=s2r.reshape(len(s2r),1)
            t2r=t2
            t2r=t2r.reshape(len(t2r),1)
            Linear_estimator2.fit(s2r, t2r)
            self.Linear_estimator2=Linear_estimator2
            
            
            
            random_estimator3 = RandomForestRegressor(random_state=0, n_estimators=500)
            random_estimator3.fit(s3,t3)
            self.random_estimator3=random_estimator3
            
            Linear_estimator3 = LinearRegression()
            
            s3r=s3.full_sq
            s3r=s3r.reshape(len(s3r),1)
            t3r=t3
            t3r=t3r.reshape(len(t3r),1)
            Linear_estimator3.fit(s3r, t3r)
            self.Linear_estimator3=Linear_estimator3
            
            
            
           
        if self.FLAG2==1:
               self.TwoClassModel=twoclass
        else :
               
                    
                d3=np.ones(len(t3))
                d4=np.ones(len(t4))*-1
                          
                X = pd.concat([s3,s4])
                y = np.concatenate([d3,d4],axis=0)
                self.y=y
                self.X=X
                #print (X)
                #self.decision_clf = svm.SVC()
                TwoClassModel = RandomForestClassifier()
                TwoClassModel.fit(X, y)
                self.TwoClassModel=TwoClassModel
                
            
        Linear_estimator4 = LinearRegression()        
        s4r=s4.full_sq
        s4r=s4r.reshape(len(s4r),1)
        t4r=t4
        t4r=t4r.reshape(len(t4r),1)
        Linear_estimator4.fit(s4r, t4r)
        self.Linear_estimator4=Linear_estimator4             

        
    def predict_price(self,test_row):
        testrow1=test_row
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
        if dum > self.highcutoff1:
            #predicted_price=5
            predicted_price1= self.random_estimator3.predict(test_row)[0] 
            
            x1 = test_row.full_sq
            x1 = x1.reshape(len(x1),1)
            
            predicted_price2= self.Linear_estimator3.predict(x1)[0]
            
        
            predicted_price=0.2*predicted_price1+ 0.8*predicted_price2
            
            
        else:
            if flag==1:
                
                predicted_price1= self.random_estimator2.predict(test_row)[0] 
                x1 = test_row.full_sq
                x1 = x1.reshape(len(x1),1)
                predicted_price2= self.Linear_estimator2.predict(x1)[0]
                predicted_price=0.5*predicted_price1+ 0.5*predicted_price2
            
            else:
                if self.FLAG2==1:
                    testrow1=testrow1.drop(['sub_area', 'full_sq' ,'product_type'],axis=1)
                    testrow1=testrow1.iloc[:,2:20]
                    for fea in testrow1.columns:
                              testrow1[fea].fillna(testrow1[fea].median(),inplace=True)
                    for fea in testrow1.columns:
                              testrow1[fea].fillna(0,inplace=True)
                              
                    pclass=self.TwoClassModel.predict(testrow1)[0]
                
                else:
                       pclass=self.TwoClassModel.predict(test_row)[0]
                       
                if pclass==-1:
                       #predicted_price= 1.5
                       x1 = test_row.full_sq
                       x1 = x1.reshape(len(x1),1)
                       predicted_price= self.Linear_estimator4.predict(x1)[0]
                else:                        
                       predicted_price1= self.random_estimator3.predict(test_row)[0] 
                       x1 = test_row.full_sq
                       x1 = x1.reshape(len(x1),1)
                       predicted_price2= self.Linear_estimator3.predict(x1)[0]
                       predicted_price=.9*predicted_price1+ .1*predicted_price2
                       
        if self.specialFlag==1:
            #if (dum >70) & (dum <600) & (predicted_price > 10):
            #     predicted_price=predicted_price + 3
            if (dum > 85)  :
                 predicted_price += 1
        #predicted_price -= .1
        return predicted_price 

gc.collect()
###############################################################

class Nagatino_Sadovniki():
    #_data = pd.DataFrame
    def __init__(self, train_rows, twoclass):
        
        #### Flags
        self.specialFlag=0 ### this is for special adjustment in predict
        self.FLAG1=1  ### this is 1 for combined s2 s3
        self.FLAG2=0 ### this is 1 for global TwoClass classifier
        
        
        
        self.highcutoff = 150
        self.highcutoff1=150
        self.lowpricecutoff =4.8

        s1=train_rows[train_rows.full_sq > self.highcutoff];
        sa=train_rows[train_rows.full_sq < self.highcutoff];
        s2=sa[(sa.state.isnull()) & (sa.life_sq.isnull())]
        s3a = sa[~((sa.state.isnull()) & (sa.life_sq.isnull()))]
        s3=s3a[s3a.price_doc>self.lowpricecutoff]
        s4=s3a[s3a.price_doc <= self.lowpricecutoff]
        t1=s1.price_doc
        t2=s2.price_doc
        t3=s3.price_doc
        t4=s4.price_doc
        s1=s1.drop(['price_doc','sub_area','product_type'],axis=1)
        s1=s1.iloc[:,2:20]
        for fea in s1.columns:
                s1[fea].fillna(s1[fea].median(), inplace=True)
        for fea in s1.columns:
                s1[fea].fillna(0, inplace=True)
        s2=s2.drop(['price_doc','sub_area','product_type'],axis=1)
        s2=s2.iloc[:,2:20]
        for fea in s2.columns:
                s2[fea].fillna(s2[fea].median(), inplace=True)
        for fea in s2.columns:
                s2[fea].fillna(0, inplace=True)              
        s3=s3.drop(['price_doc','sub_area','product_type'],axis=1)
        s3=s3.iloc[:,2:20]
        for fea in s3.columns:
                s3[fea].fillna(s3[fea].median(), inplace=True)
        for fea in s3.columns:
                s3[fea].fillna(0, inplace=True)
        s4=s4.drop(['price_doc','sub_area','product_type'],axis=1)
        s4=s4.iloc[:,2:20]
        for fea in s4.columns:
                s4[fea].fillna(s4[fea].median(), inplace=True)
        for fea in s4.columns:
                s4[fea].fillna(0, inplace=True)
        
        sa=sa.drop(['price_doc','sub_area','product_type'],axis=1)
        sa=sa.iloc[:,2:20]
        for fea in sa.columns:
                sa[fea].fillna(sa[fea].median(), inplace=True)
        for fea in sa.columns:
                sa[fea].fillna(0, inplace=True)
                
                
        if self.FLAG1==1:
           X23 = pd.concat([s2,s3]) 
           y23 = np.concatenate([t2,t3],axis=0)
           random_estimator23=RandomForestRegressor(random_state=0, n_estimators=500)
           random_estimator23.fit(X23,y23)
           self.random_estimator2=random_estimator23
           self.random_estimator3=random_estimator23
           
           X23r=X23.full_sq
           X23r=X23r.reshape(len(X23r),1)
           y23r=y23.reshape(len(y23), 1)
           Linear_estimator23=LinearRegression()
           Linear_estimator23.fit(X23r, y23r)
           
           self.Linear_estimator2=Linear_estimator23
           self.Linear_estimator3=Linear_estimator23
           
        else:
            
            random_estimator2 = RandomForestRegressor(random_state=0, n_estimators=500)
            random_estimator2.fit(s2,t2)
            self.random_estimator2=random_estimator2
            
            Linear_estimator2 = LinearRegression()
            
            s2r=s2.full_sq
            s2r=s2r.reshape(len(s2r),1)
            t2r=t2
            t2r=t2r.reshape(len(t2r),1)
            Linear_estimator2.fit(s2r, t2r)
            self.Linear_estimator2=Linear_estimator2
            
            
            
            random_estimator3 = RandomForestRegressor(random_state=0, n_estimators=500)
            random_estimator3.fit(s3,t3)
            self.random_estimator3=random_estimator3
            
            Linear_estimator3 = LinearRegression()
            
            s3r=s3.full_sq
            s3r=s3r.reshape(len(s3r),1)
            t3r=t3
            t3r=t3r.reshape(len(t3r),1)
            Linear_estimator3.fit(s3r, t3r)
            self.Linear_estimator3=Linear_estimator3
            
            
            
           
        if self.FLAG2==1:
               self.TwoClassModel=twoclass
        else :
               
                    
                d3=np.ones(len(t3))
                d4=np.ones(len(t4))*-1
                          
                X = pd.concat([s3,s4])
                y = np.concatenate([d3,d4],axis=0)
                self.y=y
                self.X=X
                #print (X)
                #self.decision_clf = svm.SVC()
                TwoClassModel = RandomForestClassifier()
                TwoClassModel.fit(X, y)
                self.TwoClassModel=TwoClassModel
                
            
        Linear_estimator4 = LinearRegression()        
        s4r=s4.full_sq
        s4r=s4r.reshape(len(s4r),1)
        t4r=t4
        t4r=t4r.reshape(len(t4r),1)
        Linear_estimator4.fit(s4r, t4r)
        self.Linear_estimator4=Linear_estimator4             

        
    def predict_price(self,test_row):
        testrow1=test_row
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
        if dum > self.highcutoff1:
            #predicted_price=5
            predicted_price1= self.random_estimator3.predict(test_row)[0] 
            
            x1 = test_row.full_sq
            x1 = x1.reshape(len(x1),1)
            
            predicted_price2= self.Linear_estimator3.predict(x1)[0]
            
        
            predicted_price=0.2*predicted_price1+ 0.8*predicted_price2
            
            
        else:
            if flag==1:
                
                predicted_price1= self.random_estimator2.predict(test_row)[0] 
                x1 = test_row.full_sq
                x1 = x1.reshape(len(x1),1)
                predicted_price2= self.Linear_estimator2.predict(x1)[0]
                predicted_price=0.5*predicted_price1+ 0.5*predicted_price2
            
            else:
                if self.FLAG2==1:
                    testrow1=testrow1.drop(['sub_area', 'full_sq' ,'product_type'],axis=1)
                    testrow1=testrow1.iloc[:,2:20]
                    for fea in testrow1.columns:
                              testrow1[fea].fillna(testrow1[fea].median(),inplace=True)
                    for fea in testrow1.columns:
                              testrow1[fea].fillna(0,inplace=True)
                              
                    pclass=self.TwoClassModel.predict(testrow1)[0]
                
                else:
                       pclass=self.TwoClassModel.predict(test_row)[0]
                       
                if pclass==-1:
                       #predicted_price= 1.5
                       x1 = test_row.full_sq
                       x1 = x1.reshape(len(x1),1)
                       predicted_price= self.Linear_estimator4.predict(x1)[0]
                else:                        
                       predicted_price1= self.random_estimator3.predict(test_row)[0] 
                       x1 = test_row.full_sq
                       x1 = x1.reshape(len(x1),1)
                       predicted_price2= self.Linear_estimator3.predict(x1)[0]
                       predicted_price=.9*predicted_price1+ .1*predicted_price2
                       
        if self.specialFlag==1:
            #if (dum >70) & (dum <600) & (predicted_price > 10):
            #     predicted_price=predicted_price + 3
            if (dum > 85)  :
                 predicted_price += 1
        #predicted_price -= .1
        return predicted_price 

gc.collect()
###############################################################

class Ramenki():
    #_data = pd.DataFrame
    def __init__(self, train_rows, twoclass):
        
        #### Flags
        self.specialFlag=0 ### this is for special adjustment in predict
        self.FLAG1=0  ### this is 1 for combined s2 s3
        self.FLAG2=0 ### this is 1 for global TwoClass classifier
        
        
        
        self.highcutoff = 150
        self.highcutoff1=150
        self.lowpricecutoff =3.2

        s1=train_rows[train_rows.full_sq > self.highcutoff];
        sa=train_rows[train_rows.full_sq < self.highcutoff];
        s2=sa[(sa.state.isnull()) & (sa.life_sq.isnull())]
        s3a = sa[~((sa.state.isnull()) & (sa.life_sq.isnull()))]
        s3=s3a[s3a.price_doc>self.lowpricecutoff]
        s4=s3a[s3a.price_doc <= self.lowpricecutoff]
        t1=s1.price_doc
        t2=s2.price_doc
        t3=s3.price_doc
        t4=s4.price_doc
        s1=s1.drop(['price_doc','sub_area','product_type'],axis=1)
        s1=s1.iloc[:,2:20]
        for fea in s1.columns:
                s1[fea].fillna(s1[fea].median(), inplace=True)
        for fea in s1.columns:
                s1[fea].fillna(0, inplace=True)
        s2=s2.drop(['price_doc','sub_area','product_type'],axis=1)
        s2=s2.iloc[:,2:20]
        for fea in s2.columns:
                s2[fea].fillna(s2[fea].median(), inplace=True)
        for fea in s2.columns:
                s2[fea].fillna(0, inplace=True)              
        s3=s3.drop(['price_doc','sub_area','product_type'],axis=1)
        s3=s3.iloc[:,2:20]
        for fea in s3.columns:
                s3[fea].fillna(s3[fea].median(), inplace=True)
        for fea in s3.columns:
                s3[fea].fillna(0, inplace=True)
        s4=s4.drop(['price_doc','sub_area','product_type'],axis=1)
        s4=s4.iloc[:,2:20]
        for fea in s4.columns:
                s4[fea].fillna(s4[fea].median(), inplace=True)
        for fea in s4.columns:
                s4[fea].fillna(0, inplace=True)
        
        sa=sa.drop(['price_doc','sub_area','product_type'],axis=1)
        sa=sa.iloc[:,2:20]
        for fea in sa.columns:
                sa[fea].fillna(sa[fea].median(), inplace=True)
        for fea in sa.columns:
                sa[fea].fillna(0, inplace=True)
                
                
        if self.FLAG1==1:
           X23 = pd.concat([s2,s3]) 
           y23 = np.concatenate([t2,t3],axis=0)
           random_estimator23=RandomForestRegressor(random_state=0, n_estimators=500)
           random_estimator23.fit(X23,y23)
           self.random_estimator2=random_estimator23
           self.random_estimator3=random_estimator23
           
           X23r=X23.full_sq
           X23r=X23r.reshape(len(X23r),1)
           y23r=y23.reshape(len(y23), 1)
           Linear_estimator23=LinearRegression()
           Linear_estimator23.fit(X23r, y23r)
           
           self.Linear_estimator2=Linear_estimator23
           self.Linear_estimator3=Linear_estimator23
           
        else:
            
            random_estimator2 = RandomForestRegressor(random_state=0, n_estimators=500)
            random_estimator2.fit(s2,t2)
            self.random_estimator2=random_estimator2
            
            Linear_estimator2 = LinearRegression()
            
            s2r=s2.full_sq
            s2r=s2r.reshape(len(s2r),1)
            t2r=t2
            t2r=t2r.reshape(len(t2r),1)
            Linear_estimator2.fit(s2r, t2r)
            self.Linear_estimator2=Linear_estimator2
            
            
            
            random_estimator3 = RandomForestRegressor(random_state=0, n_estimators=500)
            random_estimator3.fit(s3,t3)
            self.random_estimator3=random_estimator3
            
            Linear_estimator3 = LinearRegression()
            
            s3r=s3.full_sq
            s3r=s3r.reshape(len(s3r),1)
            t3r=t3
            t3r=t3r.reshape(len(t3r),1)
            Linear_estimator3.fit(s3r, t3r)
            self.Linear_estimator3=Linear_estimator3
            
            
            
           
        if self.FLAG2==1:
               self.TwoClassModel=twoclass
        else :
               
                    
                d3=np.ones(len(t3))
                d4=np.ones(len(t4))*-1
                          
                X = pd.concat([s3,s4])
                y = np.concatenate([d3,d4],axis=0)
                self.y=y
                self.X=X
                #print (X)
                #self.decision_clf = svm.SVC()
                TwoClassModel = RandomForestClassifier()
                TwoClassModel.fit(X, y)
                self.TwoClassModel=TwoClassModel
                
            
        Linear_estimator4 = LinearRegression()        
        s4r=s4.full_sq
        s4r=s4r.reshape(len(s4r),1)
        t4r=t4
        t4r=t4r.reshape(len(t4r),1)
        Linear_estimator4.fit(s4r, t4r)
        self.Linear_estimator4=Linear_estimator4             

        
    def predict_price(self,test_row):
        testrow1=test_row
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
        if dum > self.highcutoff1:
            #predicted_price=5
            predicted_price1= self.random_estimator3.predict(test_row)[0] 
            
            x1 = test_row.full_sq
            x1 = x1.reshape(len(x1),1)
            
            predicted_price2= self.Linear_estimator3.predict(x1)[0]
            
        
            predicted_price=0.2*predicted_price1+ 0.8*predicted_price2
            
            
        else:
            if flag==1:
                
                predicted_price1= self.random_estimator2.predict(test_row)[0] 
                x1 = test_row.full_sq
                x1 = x1.reshape(len(x1),1)
                predicted_price2= self.Linear_estimator2.predict(x1)[0]
                predicted_price=0.5*predicted_price1+ 0.5*predicted_price2
            
            else:
                if self.FLAG2==1:
                    testrow1=testrow1.drop(['sub_area', 'full_sq' ,'product_type'],axis=1)
                    testrow1=testrow1.iloc[:,2:20]
                    for fea in testrow1.columns:
                              testrow1[fea].fillna(testrow1[fea].median(),inplace=True)
                    for fea in testrow1.columns:
                              testrow1[fea].fillna(0,inplace=True)
                              
                    pclass=self.TwoClassModel.predict(testrow1)[0]
                
                else:
                       pclass=self.TwoClassModel.predict(test_row)[0]
                       
                if pclass==-1:
                       #predicted_price= 1.5
                       x1 = test_row.full_sq
                       x1 = x1.reshape(len(x1),1)
                       predicted_price= self.Linear_estimator4.predict(x1)[0]
                else:                        
                       predicted_price1= self.random_estimator3.predict(test_row)[0] 
                       x1 = test_row.full_sq
                       x1 = x1.reshape(len(x1),1)
                       predicted_price2= self.Linear_estimator3.predict(x1)[0]
                       predicted_price=.9*predicted_price1+ .1*predicted_price2
                       
        if self.specialFlag==1:
            #if (dum >70) & (dum <600) & (predicted_price > 10):
            #     predicted_price=predicted_price + 3
            if (dum > 85)  :
                 predicted_price += 1
        #predicted_price -= .1
        return predicted_price 

gc.collect()
###############################################################
class Bibirevo():
    #_data = pd.DataFrame
    def __init__(self, train_rows, twoclass):
        
        #### Flags
        self.specialFlag=0 ### this is for special adjustment in predict
        self.FLAG1=1  ### this is 1 for combined s2 s3
        self.FLAG2=0 ### this is 1 for global TwoClass classifier
        
        
        
        self.highcutoff = 150
        self.highcutoff1=150
        self.lowpricecutoff =3.2

        s1=train_rows[train_rows.full_sq > self.highcutoff];
        sa=train_rows[train_rows.full_sq < self.highcutoff];
        s2=sa[(sa.state.isnull()) & (sa.life_sq.isnull())]
        s3a = sa[~((sa.state.isnull()) & (sa.life_sq.isnull()))]
        s3=s3a[s3a.price_doc>self.lowpricecutoff]
        s4=s3a[s3a.price_doc <= self.lowpricecutoff]
        t1=s1.price_doc
        t2=s2.price_doc
        t3=s3.price_doc
        t4=s4.price_doc
        s1=s1.drop(['price_doc','sub_area','product_type'],axis=1)
        s1=s1.iloc[:,2:20]
        for fea in s1.columns:
                s1[fea].fillna(s1[fea].median(), inplace=True)
        for fea in s1.columns:
                s1[fea].fillna(0, inplace=True)
        s2=s2.drop(['price_doc','sub_area','product_type'],axis=1)
        s2=s2.iloc[:,2:20]
        for fea in s2.columns:
                s2[fea].fillna(s2[fea].median(), inplace=True)
        for fea in s2.columns:
                s2[fea].fillna(0, inplace=True)              
        s3=s3.drop(['price_doc','sub_area','product_type'],axis=1)
        s3=s3.iloc[:,2:20]
        for fea in s3.columns:
                s3[fea].fillna(s3[fea].median(), inplace=True)
        for fea in s3.columns:
                s3[fea].fillna(0, inplace=True)
        s4=s4.drop(['price_doc','sub_area','product_type'],axis=1)
        s4=s4.iloc[:,2:20]
        for fea in s4.columns:
                s4[fea].fillna(s4[fea].median(), inplace=True)
        for fea in s4.columns:
                s4[fea].fillna(0, inplace=True)
        
        sa=sa.drop(['price_doc','sub_area','product_type'],axis=1)
        sa=sa.iloc[:,2:20]
        for fea in sa.columns:
                sa[fea].fillna(sa[fea].median(), inplace=True)
        for fea in sa.columns:
                sa[fea].fillna(0, inplace=True)
                
                
        if self.FLAG1==1:
           X23 = pd.concat([s2,s3]) 
           y23 = np.concatenate([t2,t3],axis=0)
           random_estimator23=RandomForestRegressor(random_state=0, n_estimators=500)
           random_estimator23.fit(X23,y23)
           self.random_estimator2=random_estimator23
           self.random_estimator3=random_estimator23
           
           X23r=X23.full_sq
           X23r=X23r.reshape(len(X23r),1)
           y23r=y23.reshape(len(y23), 1)
           Linear_estimator23=LinearRegression()
           Linear_estimator23.fit(X23r, y23r)
           
           self.Linear_estimator2=Linear_estimator23
           self.Linear_estimator3=Linear_estimator23
           
        else:
            
            random_estimator2 = RandomForestRegressor(random_state=0, n_estimators=500)
            random_estimator2.fit(s2,t2)
            self.random_estimator2=random_estimator2
            
            Linear_estimator2 = LinearRegression()
            
            s2r=s2.full_sq
            s2r=s2r.reshape(len(s2r),1)
            t2r=t2
            t2r=t2r.reshape(len(t2r),1)
            Linear_estimator2.fit(s2r, t2r)
            self.Linear_estimator2=Linear_estimator2
            
            
            
            random_estimator3 = RandomForestRegressor(random_state=0, n_estimators=500)
            random_estimator3.fit(s3,t3)
            self.random_estimator3=random_estimator3
            
            Linear_estimator3 = LinearRegression()
            
            s3r=s3.full_sq
            s3r=s3r.reshape(len(s3r),1)
            t3r=t3
            t3r=t3r.reshape(len(t3r),1)
            Linear_estimator3.fit(s3r, t3r)
            self.Linear_estimator3=Linear_estimator3
            
            
            
           
        if self.FLAG2==1:
               self.TwoClassModel=twoclass
        else :
               
                    
                d3=np.ones(len(t3))
                d4=np.ones(len(t4))*-1
                          
                X = pd.concat([s3,s4])
                y = np.concatenate([d3,d4],axis=0)
                self.y=y
                self.X=X
                #print (X)
                #self.decision_clf = svm.SVC()
                TwoClassModel = RandomForestClassifier()
                TwoClassModel.fit(X, y)
                self.TwoClassModel=TwoClassModel
                
            
        Linear_estimator4 = LinearRegression()        
        s4r=s4.full_sq
        s4r=s4r.reshape(len(s4r),1)
        t4r=t4
        t4r=t4r.reshape(len(t4r),1)
        Linear_estimator4.fit(s4r, t4r)
        self.Linear_estimator4=Linear_estimator4             

        
    def predict_price(self,test_row):
        testrow1=test_row
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
        if dum > self.highcutoff1:
            #predicted_price=5
            predicted_price1= self.random_estimator3.predict(test_row)[0] 
            
            x1 = test_row.full_sq
            x1 = x1.reshape(len(x1),1)
            
            predicted_price2= self.Linear_estimator3.predict(x1)[0]
            
        
            predicted_price=0.2*predicted_price1+ 0.8*predicted_price2
            
            
        else:
            if flag==1:
                
                predicted_price1= self.random_estimator2.predict(test_row)[0] 
                x1 = test_row.full_sq
                x1 = x1.reshape(len(x1),1)
                predicted_price2= self.Linear_estimator2.predict(x1)[0]
                predicted_price=0.5*predicted_price1+ 0.5*predicted_price2
            
            else:
                if self.FLAG2==1:
                    testrow1=testrow1.drop(['sub_area', 'full_sq' ,'product_type'],axis=1)
                    testrow1=testrow1.iloc[:,2:20]
                    for fea in testrow1.columns:
                              testrow1[fea].fillna(testrow1[fea].median(),inplace=True)
                    for fea in testrow1.columns:
                              testrow1[fea].fillna(0,inplace=True)
                              
                    pclass=self.TwoClassModel.predict(testrow1)[0]
                
                else:
                       pclass=self.TwoClassModel.predict(test_row)[0]
                       
                if pclass==-1:
                       #predicted_price= 1.5
                       x1 = test_row.full_sq
                       x1 = x1.reshape(len(x1),1)
                       predicted_price= self.Linear_estimator4.predict(x1)[0]
                else:                        
                       predicted_price1= self.random_estimator3.predict(test_row)[0] 
                       x1 = test_row.full_sq
                       x1 = x1.reshape(len(x1),1)
                       predicted_price2= self.Linear_estimator3.predict(x1)[0]
                       predicted_price=.9*predicted_price1+ .1*predicted_price2
                       
        if self.specialFlag==1:
            #if (dum >70) & (dum <600) & (predicted_price > 10):
            #     predicted_price=predicted_price + 3
            if (dum > 85)  :
                 predicted_price += 1
        #predicted_price -= .1
        return predicted_price 

gc.collect()
###############################################################

class Zjablikovo():
    #_data = pd.DataFrame
    def __init__(self, train_rows, twoclass):
        
        #### Flags
        self.specialFlag=0 ### this is for special adjustment in predict
        self.FLAG1=1  ### this is 1 for combined s2 s3
        self.FLAG2=1 ### this is 1 for global TwoClass classifier
        
        
        
        self.highcutoff = 150
        self.highcutoff1=150
        self.lowpricecutoff =3.5

        s1=train_rows[train_rows.full_sq > self.highcutoff];
        sa=train_rows[train_rows.full_sq < self.highcutoff];
        s2=sa[(sa.state.isnull()) & (sa.life_sq.isnull())]
        s3a = sa[~((sa.state.isnull()) & (sa.life_sq.isnull()))]
        s3=s3a[s3a.price_doc>self.lowpricecutoff]
        s4=s3a[s3a.price_doc <= self.lowpricecutoff]
        t1=s1.price_doc
        t2=s2.price_doc
        t3=s3.price_doc
        t4=s4.price_doc
        s1=s1.drop(['price_doc','sub_area','product_type'],axis=1)
        s1=s1.iloc[:,2:20]
        for fea in s1.columns:
                s1[fea].fillna(s1[fea].median(), inplace=True)
        for fea in s1.columns:
                s1[fea].fillna(0, inplace=True)
        s2=s2.drop(['price_doc','sub_area','product_type'],axis=1)
        s2=s2.iloc[:,2:20]
        for fea in s2.columns:
                s2[fea].fillna(s2[fea].median(), inplace=True)
        for fea in s2.columns:
                s2[fea].fillna(0, inplace=True)              
        s3=s3.drop(['price_doc','sub_area','product_type'],axis=1)
        s3=s3.iloc[:,2:20]
        for fea in s3.columns:
                s3[fea].fillna(s3[fea].median(), inplace=True)
        for fea in s3.columns:
                s3[fea].fillna(0, inplace=True)
        s4=s4.drop(['price_doc','sub_area','product_type'],axis=1)
        s4=s4.iloc[:,2:20]
        for fea in s4.columns:
                s4[fea].fillna(s4[fea].median(), inplace=True)
        for fea in s4.columns:
                s4[fea].fillna(0, inplace=True)
        
        sa=sa.drop(['price_doc','sub_area','product_type'],axis=1)
        sa=sa.iloc[:,2:20]
        for fea in sa.columns:
                sa[fea].fillna(sa[fea].median(), inplace=True)
        for fea in sa.columns:
                sa[fea].fillna(0, inplace=True)
                
                
        if self.FLAG1==1:
           X23 = pd.concat([s2,s3]) 
           y23 = np.concatenate([t2,t3],axis=0)
           random_estimator23=RandomForestRegressor(random_state=0, n_estimators=500)
           random_estimator23.fit(X23,y23)
           self.random_estimator2=random_estimator23
           self.random_estimator3=random_estimator23
           
           X23r=X23.full_sq
           X23r=X23r.reshape(len(X23r),1)
           y23r=y23.reshape(len(y23), 1)
           Linear_estimator23=LinearRegression()
           Linear_estimator23.fit(X23r, y23r)
           
           self.Linear_estimator2=Linear_estimator23
           self.Linear_estimator3=Linear_estimator23
           
        else:
            
            random_estimator2 = RandomForestRegressor(random_state=0, n_estimators=500)
            random_estimator2.fit(s2,t2)
            self.random_estimator2=random_estimator2
            
            Linear_estimator2 = LinearRegression()
            
            s2r=s2.full_sq
            s2r=s2r.reshape(len(s2r),1)
            t2r=t2
            t2r=t2r.reshape(len(t2r),1)
            Linear_estimator2.fit(s2r, t2r)
            self.Linear_estimator2=Linear_estimator2
            
            
            
            random_estimator3 = RandomForestRegressor(random_state=0, n_estimators=500)
            random_estimator3.fit(s3,t3)
            self.random_estimator3=random_estimator3
            
            Linear_estimator3 = LinearRegression()
            
            s3r=s3.full_sq
            s3r=s3r.reshape(len(s3r),1)
            t3r=t3
            t3r=t3r.reshape(len(t3r),1)
            Linear_estimator3.fit(s3r, t3r)
            self.Linear_estimator3=Linear_estimator3
            
            
            
           
        if self.FLAG2==1:
               self.TwoClassModel=twoclass
        else :
               
                    
                d3=np.ones(len(t3))
                d4=np.ones(len(t4))*-1
                          
                X = pd.concat([s3,s4])
                y = np.concatenate([d3,d4],axis=0)
                self.y=y
                self.X=X
                #print (X)
                #self.decision_clf = svm.SVC()
                TwoClassModel = RandomForestClassifier()
                TwoClassModel.fit(X, y)
                self.TwoClassModel=TwoClassModel
                
            
        Linear_estimator4 = LinearRegression()        
        s4r=s4.full_sq
        s4r=s4r.reshape(len(s4r),1)
        t4r=t4
        t4r=t4r.reshape(len(t4r),1)
        Linear_estimator4.fit(s4r, t4r)
        self.Linear_estimator4=Linear_estimator4             

        
    def predict_price(self,test_row):
        testrow1=test_row
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
        if dum > self.highcutoff1:
            #predicted_price=5
            predicted_price1= self.random_estimator3.predict(test_row)[0] 
            
            x1 = test_row.full_sq
            x1 = x1.reshape(len(x1),1)
            
            predicted_price2= self.Linear_estimator3.predict(x1)[0]
            
        
            predicted_price=0.2*predicted_price1+ 0.8*predicted_price2
            
            
        else:
            if flag==1:
                
                predicted_price1= self.random_estimator2.predict(test_row)[0] 
                x1 = test_row.full_sq
                x1 = x1.reshape(len(x1),1)
                predicted_price2= self.Linear_estimator2.predict(x1)[0]
                predicted_price=0.5*predicted_price1+ 0.5*predicted_price2
            
            else:
                if self.FLAG2==1:
                    testrow1=testrow1.drop(['sub_area', 'full_sq' ,'product_type'],axis=1)
                    testrow1=testrow1.iloc[:,2:20]
                    for fea in testrow1.columns:
                              testrow1[fea].fillna(testrow1[fea].median(),inplace=True)
                    for fea in testrow1.columns:
                              testrow1[fea].fillna(0,inplace=True)
                              
                    pclass=self.TwoClassModel.predict(testrow1)[0]
                
                else:
                       pclass=self.TwoClassModel.predict(test_row)[0]
                       
                if pclass==-1:
                       #predicted_price= 1.5
                       x1 = test_row.full_sq
                       x1 = x1.reshape(len(x1),1)
                       predicted_price= self.Linear_estimator4.predict(x1)[0]
                else:                        
                       predicted_price1= self.random_estimator3.predict(test_row)[0] 
                       x1 = test_row.full_sq
                       x1 = x1.reshape(len(x1),1)
                       predicted_price2= self.Linear_estimator3.predict(x1)[0]
                       predicted_price=.9*predicted_price1+ .1*predicted_price2
                       
        if self.specialFlag==1:
            #if (dum >70) & (dum <600) & (predicted_price > 10):
            #     predicted_price=predicted_price + 3
            if (dum > 85)  :
                 predicted_price += 1
        #predicted_price -= .1
        return predicted_price 

gc.collect()
###############################################################
class Meshhanskoe():
    #_data = pd.DataFrame
    def __init__(self, train_rows, twoclass):
        
        #### Flags
        self.specialFlag=0 ### this is for special adjustment in predict
        self.FLAG1=0  ### this is 1 for combined s2 s3
        self.FLAG2=0 ### this is 1 for global TwoClass classifier
        
        
        
        self.highcutoff = 150
        self.highcutoff1=150
        self.lowpricecutoff =4.3

        s1=train_rows[train_rows.full_sq > self.highcutoff];
        sa=train_rows[train_rows.full_sq < self.highcutoff];
        s2=sa[(sa.state.isnull()) & (sa.life_sq.isnull())]
        s3a = sa[~((sa.state.isnull()) & (sa.life_sq.isnull()))]
        s3=s3a[s3a.price_doc>self.lowpricecutoff]
        s4=s3a[s3a.price_doc <= self.lowpricecutoff]
        t1=s1.price_doc
        t2=s2.price_doc
        t3=s3.price_doc
        t4=s4.price_doc
        s1=s1.drop(['price_doc','sub_area','product_type'],axis=1)
        s1=s1.iloc[:,2:20]
        for fea in s1.columns:
                s1[fea].fillna(s1[fea].median(), inplace=True)
        for fea in s1.columns:
                s1[fea].fillna(0, inplace=True)
        s2=s2.drop(['price_doc','sub_area','product_type'],axis=1)
        s2=s2.iloc[:,2:20]
        for fea in s2.columns:
                s2[fea].fillna(s2[fea].median(), inplace=True)
        for fea in s2.columns:
                s2[fea].fillna(0, inplace=True)              
        s3=s3.drop(['price_doc','sub_area','product_type'],axis=1)
        s3=s3.iloc[:,2:20]
        for fea in s3.columns:
                s3[fea].fillna(s3[fea].median(), inplace=True)
        for fea in s3.columns:
                s3[fea].fillna(0, inplace=True)
        s4=s4.drop(['price_doc','sub_area','product_type'],axis=1)
        s4=s4.iloc[:,2:20]
        for fea in s4.columns:
                s4[fea].fillna(s4[fea].median(), inplace=True)
        for fea in s4.columns:
                s4[fea].fillna(0, inplace=True)
        
        sa=sa.drop(['price_doc','sub_area','product_type'],axis=1)
        sa=sa.iloc[:,2:20]
        for fea in sa.columns:
                sa[fea].fillna(sa[fea].median(), inplace=True)
        for fea in sa.columns:
                sa[fea].fillna(0, inplace=True)
                
                
        if self.FLAG1==1:
           X23 = pd.concat([s2,s3]) 
           y23 = np.concatenate([t2,t3],axis=0)
           random_estimator23=RandomForestRegressor(random_state=0, n_estimators=500)
           random_estimator23.fit(X23,y23)
           self.random_estimator2=random_estimator23
           self.random_estimator3=random_estimator23
           
           X23r=X23.full_sq
           X23r=X23r.reshape(len(X23r),1)
           y23r=y23.reshape(len(y23), 1)
           Linear_estimator23=LinearRegression()
           Linear_estimator23.fit(X23r, y23r)
           
           self.Linear_estimator2=Linear_estimator23
           self.Linear_estimator3=Linear_estimator23
           
        else:
            
            random_estimator2 = RandomForestRegressor(random_state=0, n_estimators=500)
            random_estimator2.fit(s2,t2)
            self.random_estimator2=random_estimator2
            
            Linear_estimator2 = LinearRegression()
            
            s2r=s2.full_sq
            s2r=s2r.reshape(len(s2r),1)
            t2r=t2
            t2r=t2r.reshape(len(t2r),1)
            Linear_estimator2.fit(s2r, t2r)
            self.Linear_estimator2=Linear_estimator2
            
            
            
            random_estimator3 = RandomForestRegressor(random_state=0, n_estimators=500)
            random_estimator3.fit(s3,t3)
            self.random_estimator3=random_estimator3
            
            Linear_estimator3 = LinearRegression()
            
            s3r=s3.full_sq
            s3r=s3r.reshape(len(s3r),1)
            t3r=t3
            t3r=t3r.reshape(len(t3r),1)
            Linear_estimator3.fit(s3r, t3r)
            self.Linear_estimator3=Linear_estimator3
            
            
            
           
        if self.FLAG2==1:
               self.TwoClassModel=twoclass
        else :
               
                    
                d3=np.ones(len(t3))
                d4=np.ones(len(t4))*-1
                          
                X = pd.concat([s3,s4])
                y = np.concatenate([d3,d4],axis=0)
                self.y=y
                self.X=X
                #print (X)
                #self.decision_clf = svm.SVC()
                TwoClassModel = RandomForestClassifier()
                TwoClassModel.fit(X, y)
                self.TwoClassModel=TwoClassModel
                
            
        Linear_estimator4 = LinearRegression()        
        s4r=s4.full_sq
        s4r=s4r.reshape(len(s4r),1)
        t4r=t4
        t4r=t4r.reshape(len(t4r),1)
        Linear_estimator4.fit(s4r, t4r)
        self.Linear_estimator4=Linear_estimator4             

        
    def predict_price(self,test_row):
        testrow1=test_row
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
        if dum > self.highcutoff1:
            #predicted_price=5
            predicted_price1= self.random_estimator3.predict(test_row)[0] 
            
            x1 = test_row.full_sq
            x1 = x1.reshape(len(x1),1)
            
            predicted_price2= self.Linear_estimator3.predict(x1)[0]
            
        
            predicted_price=0.2*predicted_price1+ 0.8*predicted_price2
            
            
        else:
            if flag==1:
                
                predicted_price1= self.random_estimator2.predict(test_row)[0] 
                x1 = test_row.full_sq
                x1 = x1.reshape(len(x1),1)
                predicted_price2= self.Linear_estimator2.predict(x1)[0]
                predicted_price=0.5*predicted_price1+ 0.5*predicted_price2
            
            else:
                if self.FLAG2==1:
                    testrow1=testrow1.drop(['sub_area', 'full_sq' ,'product_type'],axis=1)
                    testrow1=testrow1.iloc[:,2:20]
                    for fea in testrow1.columns:
                              testrow1[fea].fillna(testrow1[fea].median(),inplace=True)
                    for fea in testrow1.columns:
                              testrow1[fea].fillna(0,inplace=True)
                              
                    pclass=self.TwoClassModel.predict(testrow1)[0]
                
                else:
                       pclass=self.TwoClassModel.predict(test_row)[0]
                       
                if pclass==-1:
                       #predicted_price= 1.5
                       x1 = test_row.full_sq
                       x1 = x1.reshape(len(x1),1)
                       predicted_price= self.Linear_estimator4.predict(x1)[0]
                else:                        
                       predicted_price1= self.random_estimator3.predict(test_row)[0] 
                       x1 = test_row.full_sq
                       x1 = x1.reshape(len(x1),1)
                       predicted_price2= self.Linear_estimator3.predict(x1)[0]
                       predicted_price=.9*predicted_price1+ .1*predicted_price2
                       
        if self.specialFlag==1:
            #if (dum >70) & (dum <600) & (predicted_price > 10):
            #     predicted_price=predicted_price + 3
            if (dum > 85)  :
                 predicted_price += 1
        #predicted_price -= .1
        return predicted_price 

gc.collect()
###############################################################
class Chertanovo_Juzhnoel():
    #_data = pd.DataFrame
    def __init__(self, train_rows, twoclass):
        
        #### Flags
        self.specialFlag=1 ### this is for special adjustment in predict
        self.FLAG1=0  ### this is 1 for combined s2 s3
        self.FLAG2=0 ### this is 1 for global TwoClass classifier
        
        
        
        self.highcutoff = 150
        self.highcutoff1=150
        self.lowpricecutoff =3.2

        s1=train_rows[train_rows.full_sq > self.highcutoff];
        sa=train_rows[train_rows.full_sq < self.highcutoff];
        s2=sa[(sa.state.isnull()) & (sa.life_sq.isnull())]
        s3a = sa[~((sa.state.isnull()) & (sa.life_sq.isnull()))]
        s3=s3a[s3a.price_doc>self.lowpricecutoff]
        s4=s3a[s3a.price_doc <= self.lowpricecutoff]
        t1=s1.price_doc
        t2=s2.price_doc
        t3=s3.price_doc
        t4=s4.price_doc
        s1=s1.drop(['price_doc','sub_area','product_type'],axis=1)
        s1=s1.iloc[:,2:20]
        for fea in s1.columns:
                s1[fea].fillna(s1[fea].median(), inplace=True)
        for fea in s1.columns:
                s1[fea].fillna(0, inplace=True)
        s2=s2.drop(['price_doc','sub_area','product_type'],axis=1)
        s2=s2.iloc[:,2:20]
        for fea in s2.columns:
                s2[fea].fillna(s2[fea].median(), inplace=True)
        for fea in s2.columns:
                s2[fea].fillna(0, inplace=True)              
        s3=s3.drop(['price_doc','sub_area','product_type'],axis=1)
        s3=s3.iloc[:,2:20]
        for fea in s3.columns:
                s3[fea].fillna(s3[fea].median(), inplace=True)
        for fea in s3.columns:
                s3[fea].fillna(0, inplace=True)
        s4=s4.drop(['price_doc','sub_area','product_type'],axis=1)
        s4=s4.iloc[:,2:20]
        for fea in s4.columns:
                s4[fea].fillna(s4[fea].median(), inplace=True)
        for fea in s4.columns:
                s4[fea].fillna(0, inplace=True)
        
        sa=sa.drop(['price_doc','sub_area','product_type'],axis=1)
        sa=sa.iloc[:,2:20]
        for fea in sa.columns:
                sa[fea].fillna(sa[fea].median(), inplace=True)
        for fea in sa.columns:
                sa[fea].fillna(0, inplace=True)
                
                
        if self.FLAG1==1:
           X23 = pd.concat([s2,s3]) 
           y23 = np.concatenate([t2,t3],axis=0)
           random_estimator23=RandomForestRegressor(random_state=0, n_estimators=500)
           random_estimator23.fit(X23,y23)
           self.random_estimator2=random_estimator23
           self.random_estimator3=random_estimator23
           
           X23r=X23.full_sq
           X23r=X23r.reshape(len(X23r),1)
           y23r=y23.reshape(len(y23), 1)
           Linear_estimator23=LinearRegression()
           Linear_estimator23.fit(X23r, y23r)
           
           self.Linear_estimator2=Linear_estimator23
           self.Linear_estimator3=Linear_estimator23
           
        else:
            
            random_estimator2 = RandomForestRegressor(random_state=0, n_estimators=500)
            random_estimator2.fit(s2,t2)
            self.random_estimator2=random_estimator2
            
            Linear_estimator2 = LinearRegression()
            
            s2r=s2.full_sq
            s2r=s2r.reshape(len(s2r),1)
            t2r=t2
            t2r=t2r.reshape(len(t2r),1)
            Linear_estimator2.fit(s2r, t2r)
            self.Linear_estimator2=Linear_estimator2
            
            
            
            random_estimator3 = RandomForestRegressor(random_state=0, n_estimators=500)
            random_estimator3.fit(s3,t3)
            self.random_estimator3=random_estimator3
            
            Linear_estimator3 = LinearRegression()
            
            s3r=s3.full_sq
            s3r=s3r.reshape(len(s3r),1)
            t3r=t3
            t3r=t3r.reshape(len(t3r),1)
            Linear_estimator3.fit(s3r, t3r)
            self.Linear_estimator3=Linear_estimator3
            
            
            
           
        if self.FLAG2==1:
               self.TwoClassModel=twoclass
        else :
               
                    
                d3=np.ones(len(t3))
                d4=np.ones(len(t4))*-1
                          
                X = pd.concat([s3,s4])
                y = np.concatenate([d3,d4],axis=0)
                self.y=y
                self.X=X
                #print (X)
                #self.decision_clf = svm.SVC()
                TwoClassModel = RandomForestClassifier()
                TwoClassModel.fit(X, y)
                self.TwoClassModel=TwoClassModel
                
            
        Linear_estimator4 = LinearRegression()        
        s4r=s4.full_sq
        s4r=s4r.reshape(len(s4r),1)
        t4r=t4
        t4r=t4r.reshape(len(t4r),1)
        Linear_estimator4.fit(s4r, t4r)
        self.Linear_estimator4=Linear_estimator4             

        
    def predict_price(self,test_row):
        testrow1=test_row
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
        if dum > self.highcutoff1:
            #predicted_price=5
            predicted_price1= self.random_estimator3.predict(test_row)[0] 
            
            x1 = test_row.full_sq
            x1 = x1.reshape(len(x1),1)
            
            predicted_price2= self.Linear_estimator3.predict(x1)[0]
            
        
            predicted_price=0.2*predicted_price1+ 0.8*predicted_price2
            
            
        else:
            if flag==1:
                
                predicted_price1= self.random_estimator2.predict(test_row)[0] 
                x1 = test_row.full_sq
                x1 = x1.reshape(len(x1),1)
                predicted_price2= self.Linear_estimator2.predict(x1)[0]
                predicted_price=0.5*predicted_price1+ 0.5*predicted_price2
            
            else:
                if self.FLAG2==1:
                    testrow1=testrow1.drop(['sub_area', 'full_sq' ,'product_type'],axis=1)
                    testrow1=testrow1.iloc[:,2:20]
                    for fea in testrow1.columns:
                              testrow1[fea].fillna(testrow1[fea].median(),inplace=True)
                    for fea in testrow1.columns:
                              testrow1[fea].fillna(0,inplace=True)
                              
                    pclass=self.TwoClassModel.predict(testrow1)[0]
                
                else:
                       pclass=self.TwoClassModel.predict(test_row)[0]
                       
                if pclass==-1:
                       #predicted_price= 1.5
                       x1 = test_row.full_sq
                       x1 = x1.reshape(len(x1),1)
                       predicted_price= self.Linear_estimator4.predict(x1)[0]
                else:                        
                       predicted_price1= self.random_estimator3.predict(test_row)[0] 
                       x1 = test_row.full_sq
                       x1 = x1.reshape(len(x1),1)
                       predicted_price2= self.Linear_estimator3.predict(x1)[0]
                       predicted_price=.9*predicted_price1+ .1*predicted_price2
                       
        if self.specialFlag==1:
            #if (dum >70) & (dum <600) & (predicted_price > 10):
            #     predicted_price=predicted_price + 3
            if (dum < 26)  :
                 predicted_price -= 2
        #predicted_price -= .1
        return predicted_price 

gc.collect()
###############################################################


class Danilovskoe():
    #_data = pd.DataFrame
    def __init__(self, train_rows, twoclass):
        
        #### Flags
        self.specialFlag=0  ### this is for special adjustment in predict
        self.FLAG1=1  ### this is 1 for combined s2 s3
        self.FLAG2=0 ### this is 1 for global TwoClass classifier
        
        
        
        self.highcutoff = 150
        self.highcutoff1=150
        self.lowpricecutoff =3.2

        s1=train_rows[train_rows.full_sq > self.highcutoff];
        sa=train_rows[train_rows.full_sq < self.highcutoff];
        s2=sa[(sa.state.isnull()) & (sa.life_sq.isnull())]
        s3a = sa[~((sa.state.isnull()) & (sa.life_sq.isnull()))]
        s3=s3a[s3a.price_doc>self.lowpricecutoff]
        s4=s3a[s3a.price_doc <= self.lowpricecutoff]
        t1=s1.price_doc
        t2=s2.price_doc
        t3=s3.price_doc
        t4=s4.price_doc
        s1=s1.drop(['price_doc','sub_area','product_type'],axis=1)
        s1=s1.iloc[:,2:20]
        for fea in s1.columns:
                s1[fea].fillna(s1[fea].median(), inplace=True)
        for fea in s1.columns:
                s1[fea].fillna(0, inplace=True)
        s2=s2.drop(['price_doc','sub_area','product_type'],axis=1)
        s2=s2.iloc[:,2:20]
        for fea in s2.columns:
                s2[fea].fillna(s2[fea].median(), inplace=True)
        for fea in s2.columns:
                s2[fea].fillna(0, inplace=True)              
        s3=s3.drop(['price_doc','sub_area','product_type'],axis=1)
        s3=s3.iloc[:,2:20]
        for fea in s3.columns:
                s3[fea].fillna(s3[fea].median(), inplace=True)
        for fea in s3.columns:
                s3[fea].fillna(0, inplace=True)
        s4=s4.drop(['price_doc','sub_area','product_type'],axis=1)
        s4=s4.iloc[:,2:20]
        for fea in s4.columns:
                s4[fea].fillna(s4[fea].median(), inplace=True)
        for fea in s4.columns:
                s4[fea].fillna(0, inplace=True)
        
        sa=sa.drop(['price_doc','sub_area','product_type'],axis=1)
        sa=sa.iloc[:,2:20]
        for fea in sa.columns:
                sa[fea].fillna(sa[fea].median(), inplace=True)
        for fea in sa.columns:
                sa[fea].fillna(0, inplace=True)
                
                
        if self.FLAG1==1:
           X23 = pd.concat([s2,s3]) 
           y23 = np.concatenate([t2,t3],axis=0)
           random_estimator23=RandomForestRegressor(random_state=0, n_estimators=500)
           random_estimator23.fit(X23,y23)
           self.random_estimator2=random_estimator23
           self.random_estimator3=random_estimator23
           
           X23r=X23.full_sq
           X23r=X23r.reshape(len(X23r),1)
           y23r=y23.reshape(len(y23), 1)
           Linear_estimator23=LinearRegression()
           Linear_estimator23.fit(X23r, y23r)
           
           self.Linear_estimator2=Linear_estimator23
           self.Linear_estimator3=Linear_estimator23
           
        else:
            
            random_estimator2 = RandomForestRegressor(random_state=0, n_estimators=500)
            random_estimator2.fit(s2,t2)
            self.random_estimator2=random_estimator2
            
            Linear_estimator2 = LinearRegression()
            
            s2r=s2.full_sq
            s2r=s2r.reshape(len(s2r),1)
            t2r=t2
            t2r=t2r.reshape(len(t2r),1)
            Linear_estimator2.fit(s2r, t2r)
            self.Linear_estimator2=Linear_estimator2
            
            
            
            random_estimator3 = RandomForestRegressor(random_state=0, n_estimators=500)
            random_estimator3.fit(s3,t3)
            self.random_estimator3=random_estimator3
            
            Linear_estimator3 = LinearRegression()
            
            s3r=s3.full_sq
            s3r=s3r.reshape(len(s3r),1)
            t3r=t3
            t3r=t3r.reshape(len(t3r),1)
            Linear_estimator3.fit(s3r, t3r)
            self.Linear_estimator3=Linear_estimator3
            
            
            
           
        if self.FLAG2==1:
               self.TwoClassModel=twoclass
        else :
               
                    
                d3=np.ones(len(t3))
                d4=np.ones(len(t4))*-1
                          
                X = pd.concat([s3,s4])
                y = np.concatenate([d3,d4],axis=0)
                self.y=y
                self.X=X
                #print (X)
                #self.decision_clf = svm.SVC()
                TwoClassModel = RandomForestClassifier()
                TwoClassModel.fit(X, y)
                self.TwoClassModel=TwoClassModel
                
            
        Linear_estimator4 = LinearRegression()        
        s4r=s4.full_sq
        s4r=s4r.reshape(len(s4r),1)
        t4r=t4
        t4r=t4r.reshape(len(t4r),1)
        Linear_estimator4.fit(s4r, t4r)
        self.Linear_estimator4=Linear_estimator4             

        
    def predict_price(self,test_row):
        testrow1=test_row
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
        if dum > self.highcutoff1:
            #predicted_price=5
            predicted_price1= self.random_estimator3.predict(test_row)[0] 
            
            x1 = test_row.full_sq
            x1 = x1.reshape(len(x1),1)
            
            predicted_price2= self.Linear_estimator3.predict(x1)[0]
            
        
            predicted_price=0.2*predicted_price1+ 0.8*predicted_price2
            
            
        else:
            if flag==1:
                
                predicted_price1= self.random_estimator2.predict(test_row)[0] 
                x1 = test_row.full_sq
                x1 = x1.reshape(len(x1),1)
                predicted_price2= self.Linear_estimator2.predict(x1)[0]
                predicted_price=0.5*predicted_price1+ 0.5*predicted_price2
            
            else:
                if self.FLAG2==1:
                    testrow1=testrow1.drop(['sub_area', 'full_sq' ,'product_type'],axis=1)
                    testrow1=testrow1.iloc[:,2:20]
                    for fea in testrow1.columns:
                              testrow1[fea].fillna(testrow1[fea].median(),inplace=True)
                    for fea in testrow1.columns:
                              testrow1[fea].fillna(0,inplace=True)
                              
                    pclass=self.TwoClassModel.predict(testrow1)[0]
                
                else:
                       pclass=self.TwoClassModel.predict(test_row)[0]
                       
                if pclass==-1:
                       #predicted_price= 1.5
                       x1 = test_row.full_sq
                       x1 = x1.reshape(len(x1),1)
                       predicted_price= self.Linear_estimator4.predict(x1)[0]
                else:                        
                       predicted_price1= self.random_estimator3.predict(test_row)[0] 
                       x1 = test_row.full_sq
                       x1 = x1.reshape(len(x1),1)
                       predicted_price2= self.Linear_estimator3.predict(x1)[0]
                       predicted_price=.9*predicted_price1+ .1*predicted_price2
                       
        if self.specialFlag==1:
            #if (dum >70) & (dum <600) & (predicted_price > 10):
            #     predicted_price=predicted_price + 3
            if (dum < 26)  :
                 predicted_price -= 2
        #predicted_price -= .1
        return predicted_price 

gc.collect()
###############################################################
class Hovrino():
    #_data = pd.DataFrame
    def __init__(self, train_rows, twoclass):
        
        #### Flags
        self.specialFlag=0  ### this is for special adjustment in predict
        self.FLAG1=1  ### this is 1 for combined s2 s3
        self.FLAG2=0 ### this is 1 for global TwoClass classifier
        
        
        
        self.highcutoff = 150
        self.highcutoff1=150
        self.lowpricecutoff =3.2

        s1=train_rows[train_rows.full_sq > self.highcutoff];
        sa=train_rows[train_rows.full_sq < self.highcutoff];
        s2=sa[(sa.state.isnull()) & (sa.life_sq.isnull())]
        s3a = sa[~((sa.state.isnull()) & (sa.life_sq.isnull()))]
        s3=s3a[s3a.price_doc>self.lowpricecutoff]
        s4=s3a[s3a.price_doc <= self.lowpricecutoff]
        t1=s1.price_doc
        t2=s2.price_doc
        t3=s3.price_doc
        t4=s4.price_doc
        s1=s1.drop(['price_doc','sub_area','product_type'],axis=1)
        s1=s1.iloc[:,2:20]
        for fea in s1.columns:
                s1[fea].fillna(s1[fea].median(), inplace=True)
        for fea in s1.columns:
                s1[fea].fillna(0, inplace=True)
        s2=s2.drop(['price_doc','sub_area','product_type'],axis=1)
        s2=s2.iloc[:,2:20]
        for fea in s2.columns:
                s2[fea].fillna(s2[fea].median(), inplace=True)
        for fea in s2.columns:
                s2[fea].fillna(0, inplace=True)              
        s3=s3.drop(['price_doc','sub_area','product_type'],axis=1)
        s3=s3.iloc[:,2:20]
        for fea in s3.columns:
                s3[fea].fillna(s3[fea].median(), inplace=True)
        for fea in s3.columns:
                s3[fea].fillna(0, inplace=True)
        s4=s4.drop(['price_doc','sub_area','product_type'],axis=1)
        s4=s4.iloc[:,2:20]
        for fea in s4.columns:
                s4[fea].fillna(s4[fea].median(), inplace=True)
        for fea in s4.columns:
                s4[fea].fillna(0, inplace=True)
        
        sa=sa.drop(['price_doc','sub_area','product_type'],axis=1)
        sa=sa.iloc[:,2:20]
        for fea in sa.columns:
                sa[fea].fillna(sa[fea].median(), inplace=True)
        for fea in sa.columns:
                sa[fea].fillna(0, inplace=True)
                
                
        if self.FLAG1==1:
           X23 = pd.concat([s2,s3]) 
           y23 = np.concatenate([t2,t3],axis=0)
           random_estimator23=RandomForestRegressor(random_state=0, n_estimators=500)
           random_estimator23.fit(X23,y23)
           self.random_estimator2=random_estimator23
           self.random_estimator3=random_estimator23
           
           X23r=X23.full_sq
           X23r=X23r.reshape(len(X23r),1)
           y23r=y23.reshape(len(y23), 1)
           Linear_estimator23=LinearRegression()
           Linear_estimator23.fit(X23r, y23r)
           
           self.Linear_estimator2=Linear_estimator23
           self.Linear_estimator3=Linear_estimator23
           
        else:
            
            random_estimator2 = RandomForestRegressor(random_state=0, n_estimators=500)
            random_estimator2.fit(s2,t2)
            self.random_estimator2=random_estimator2
            
            Linear_estimator2 = LinearRegression()
            
            s2r=s2.full_sq
            s2r=s2r.reshape(len(s2r),1)
            t2r=t2
            t2r=t2r.reshape(len(t2r),1)
            Linear_estimator2.fit(s2r, t2r)
            self.Linear_estimator2=Linear_estimator2
            
            
            
            random_estimator3 = RandomForestRegressor(random_state=0, n_estimators=500)
            random_estimator3.fit(s3,t3)
            self.random_estimator3=random_estimator3
            
            Linear_estimator3 = LinearRegression()
            
            s3r=s3.full_sq
            s3r=s3r.reshape(len(s3r),1)
            t3r=t3
            t3r=t3r.reshape(len(t3r),1)
            Linear_estimator3.fit(s3r, t3r)
            self.Linear_estimator3=Linear_estimator3
            
            
            
           
        if self.FLAG2==1:
               self.TwoClassModel=twoclass
        else :
               
                    
                d3=np.ones(len(t3))
                d4=np.ones(len(t4))*-1
                          
                X = pd.concat([s3,s4])
                y = np.concatenate([d3,d4],axis=0)
                self.y=y
                self.X=X
                #print (X)
                #self.decision_clf = svm.SVC()
                TwoClassModel = RandomForestClassifier()
                TwoClassModel.fit(X, y)
                self.TwoClassModel=TwoClassModel
                
            
        Linear_estimator4 = LinearRegression()        
        s4r=s4.full_sq
        s4r=s4r.reshape(len(s4r),1)
        t4r=t4
        t4r=t4r.reshape(len(t4r),1)
        Linear_estimator4.fit(s4r, t4r)
        self.Linear_estimator4=Linear_estimator4             

        
    def predict_price(self,test_row):
        testrow1=test_row
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
        if dum > self.highcutoff1:
            #predicted_price=5
            predicted_price1= self.random_estimator3.predict(test_row)[0] 
            
            x1 = test_row.full_sq
            x1 = x1.reshape(len(x1),1)
            
            predicted_price2= self.Linear_estimator3.predict(x1)[0]
            
        
            predicted_price=0.2*predicted_price1+ 0.8*predicted_price2
            
            
        else:
            if flag==1:
                
                predicted_price1= self.random_estimator2.predict(test_row)[0] 
                x1 = test_row.full_sq
                x1 = x1.reshape(len(x1),1)
                predicted_price2= self.Linear_estimator2.predict(x1)[0]
                predicted_price=0.5*predicted_price1+ 0.5*predicted_price2
            
            else:
                if self.FLAG2==1:
                    testrow1=testrow1.drop(['sub_area', 'full_sq' ,'product_type'],axis=1)
                    testrow1=testrow1.iloc[:,2:20]
                    for fea in testrow1.columns:
                              testrow1[fea].fillna(testrow1[fea].median(),inplace=True)
                    for fea in testrow1.columns:
                              testrow1[fea].fillna(0,inplace=True)
                              
                    pclass=self.TwoClassModel.predict(testrow1)[0]
                
                else:
                       pclass=self.TwoClassModel.predict(test_row)[0]
                       
                if pclass==-1:
                       #predicted_price= 1.5
                       x1 = test_row.full_sq
                       x1 = x1.reshape(len(x1),1)
                       predicted_price= self.Linear_estimator4.predict(x1)[0]
                else:                        
                       predicted_price1= self.random_estimator3.predict(test_row)[0] 
                       x1 = test_row.full_sq
                       x1 = x1.reshape(len(x1),1)
                       predicted_price2= self.Linear_estimator3.predict(x1)[0]
                       predicted_price=.9*predicted_price1+ .1*predicted_price2
                       
        if self.specialFlag==1:
            #if (dum >70) & (dum <600) & (predicted_price > 10):
            #     predicted_price=predicted_price + 3
            if (dum < 26)  :
                 predicted_price -= 2
        #predicted_price -= .1
        return predicted_price 

gc.collect()
###############################################################

class Vostochnoe_Degunino():
    #_data = pd.DataFrame
    def __init__(self, train_rows, twoclass):
        
        #### Flags
        self.specialFlag=0  ### this is for special adjustment in predict
        self.FLAG1=1  ### this is 1 for combined s2 s3
        self.FLAG2=1 ### this is 1 for global TwoClass classifier
        
        
        
        self.highcutoff = 150
        self.highcutoff1=150
        self.lowpricecutoff =4.2

        s1=train_rows[train_rows.full_sq > self.highcutoff];
        sa=train_rows[train_rows.full_sq < self.highcutoff];
        s2=sa[(sa.state.isnull()) & (sa.life_sq.isnull())]
        s3a = sa[~((sa.state.isnull()) & (sa.life_sq.isnull()))]
        s3=s3a[s3a.price_doc>self.lowpricecutoff]
        s4=s3a[s3a.price_doc <= self.lowpricecutoff]
        t1=s1.price_doc
        t2=s2.price_doc
        t3=s3.price_doc
        t4=s4.price_doc
        s1=s1.drop(['price_doc','sub_area','product_type'],axis=1)
        s1=s1.iloc[:,2:20]
        for fea in s1.columns:
                s1[fea].fillna(s1[fea].median(), inplace=True)
        for fea in s1.columns:
                s1[fea].fillna(0, inplace=True)
        s2=s2.drop(['price_doc','sub_area','product_type'],axis=1)
        s2=s2.iloc[:,2:20]
        for fea in s2.columns:
                s2[fea].fillna(s2[fea].median(), inplace=True)
        for fea in s2.columns:
                s2[fea].fillna(0, inplace=True)              
        s3=s3.drop(['price_doc','sub_area','product_type'],axis=1)
        s3=s3.iloc[:,2:20]
        for fea in s3.columns:
                s3[fea].fillna(s3[fea].median(), inplace=True)
        for fea in s3.columns:
                s3[fea].fillna(0, inplace=True)
        s4=s4.drop(['price_doc','sub_area','product_type'],axis=1)
        s4=s4.iloc[:,2:20]
        for fea in s4.columns:
                s4[fea].fillna(s4[fea].median(), inplace=True)
        for fea in s4.columns:
                s4[fea].fillna(0, inplace=True)
        
        sa=sa.drop(['price_doc','sub_area','product_type'],axis=1)
        sa=sa.iloc[:,2:20]
        for fea in sa.columns:
                sa[fea].fillna(sa[fea].median(), inplace=True)
        for fea in sa.columns:
                sa[fea].fillna(0, inplace=True)
                
                
        if self.FLAG1==1:
           X23 = pd.concat([s2,s3]) 
           y23 = np.concatenate([t2,t3],axis=0)
           random_estimator23=RandomForestRegressor(random_state=0, n_estimators=500)
           random_estimator23.fit(X23,y23)
           self.random_estimator2=random_estimator23
           self.random_estimator3=random_estimator23
           
           X23r=X23.full_sq
           X23r=X23r.reshape(len(X23r),1)
           y23r=y23.reshape(len(y23), 1)
           Linear_estimator23=LinearRegression()
           Linear_estimator23.fit(X23r, y23r)
           
           self.Linear_estimator2=Linear_estimator23
           self.Linear_estimator3=Linear_estimator23
           
        else:
            
            random_estimator2 = RandomForestRegressor(random_state=0, n_estimators=500)
            random_estimator2.fit(s2,t2)
            self.random_estimator2=random_estimator2
            
            Linear_estimator2 = LinearRegression()
            
            s2r=s2.full_sq
            s2r=s2r.reshape(len(s2r),1)
            t2r=t2
            t2r=t2r.reshape(len(t2r),1)
            Linear_estimator2.fit(s2r, t2r)
            self.Linear_estimator2=Linear_estimator2
            
            
            
            random_estimator3 = RandomForestRegressor(random_state=0, n_estimators=500)
            random_estimator3.fit(s3,t3)
            self.random_estimator3=random_estimator3
            
            Linear_estimator3 = LinearRegression()
            
            s3r=s3.full_sq
            s3r=s3r.reshape(len(s3r),1)
            t3r=t3
            t3r=t3r.reshape(len(t3r),1)
            Linear_estimator3.fit(s3r, t3r)
            self.Linear_estimator3=Linear_estimator3
            
            
            
           
        if self.FLAG2==1:
               self.TwoClassModel=twoclass
        else :
               
                    
                d3=np.ones(len(t3))
                d4=np.ones(len(t4))*-1
                          
                X = pd.concat([s3,s4])
                y = np.concatenate([d3,d4],axis=0)
                self.y=y
                self.X=X
                #print (X)
                #self.decision_clf = svm.SVC()
                TwoClassModel = RandomForestClassifier()
                TwoClassModel.fit(X, y)
                self.TwoClassModel=TwoClassModel
                
            
        Linear_estimator4 = LinearRegression()        
        s4r=s4.full_sq
        s4r=s4r.reshape(len(s4r),1)
        t4r=t4
        t4r=t4r.reshape(len(t4r),1)
        Linear_estimator4.fit(s4r, t4r)
        self.Linear_estimator4=Linear_estimator4             

        
    def predict_price(self,test_row):
        testrow1=test_row
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
        if dum > self.highcutoff1:
            #predicted_price=5
            predicted_price1= self.random_estimator3.predict(test_row)[0] 
            
            x1 = test_row.full_sq
            x1 = x1.reshape(len(x1),1)
            
            predicted_price2= self.Linear_estimator3.predict(x1)[0]
            
        
            predicted_price=0.2*predicted_price1+ 0.8*predicted_price2
            
            
        else:
            if flag==1:
                
                predicted_price1= self.random_estimator2.predict(test_row)[0] 
                x1 = test_row.full_sq
                x1 = x1.reshape(len(x1),1)
                predicted_price2= self.Linear_estimator2.predict(x1)[0]
                predicted_price=0.5*predicted_price1+ 0.5*predicted_price2
            
            else:
                if self.FLAG2==1:
                    testrow1=testrow1.drop(['sub_area', 'full_sq' ,'product_type'],axis=1)
                    testrow1=testrow1.iloc[:,2:20]
                    for fea in testrow1.columns:
                              testrow1[fea].fillna(testrow1[fea].median(),inplace=True)
                    for fea in testrow1.columns:
                              testrow1[fea].fillna(0,inplace=True)
                              
                    pclass=self.TwoClassModel.predict(testrow1)[0]
                
                else:
                       pclass=self.TwoClassModel.predict(test_row)[0]
                       
                if pclass==-1:
                       #predicted_price= 1.5
                       x1 = test_row.full_sq
                       x1 = x1.reshape(len(x1),1)
                       predicted_price= self.Linear_estimator4.predict(x1)[0]
                else:                        
                       predicted_price1= self.random_estimator3.predict(test_row)[0] 
                       x1 = test_row.full_sq
                       x1 = x1.reshape(len(x1),1)
                       predicted_price2= self.Linear_estimator3.predict(x1)[0]
                       predicted_price=.9*predicted_price1+ .1*predicted_price2
                       
        if self.specialFlag==1:
            #if (dum >70) & (dum <600) & (predicted_price > 10):
            #     predicted_price=predicted_price + 3
            if (dum < 26)  :
                 predicted_price -= 2
        #predicted_price -= .1
        return predicted_price 

gc.collect()
###############################################################


class Novokosino():
    #_data = pd.DataFrame
    def __init__(self, train_rows, twoclass):
        
        #### Flags
        self.specialFlag=0  ### this is for special adjustment in predict
        self.FLAG1=1  ### this is 1 for combined s2 s3
        self.FLAG2=1 ### this is 1 for global TwoClass classifier
        
        
        
        self.highcutoff = 150
        self.highcutoff1=150
        self.lowpricecutoff =4.2

        s1=train_rows[train_rows.full_sq > self.highcutoff];
        sa=train_rows[train_rows.full_sq < self.highcutoff];
        s2=sa[(sa.state.isnull()) & (sa.life_sq.isnull())]
        s3a = sa[~((sa.state.isnull()) & (sa.life_sq.isnull()))]
        s3=s3a[s3a.price_doc>self.lowpricecutoff]
        s4=s3a[s3a.price_doc <= self.lowpricecutoff]
        t1=s1.price_doc
        t2=s2.price_doc
        t3=s3.price_doc
        t4=s4.price_doc
        s1=s1.drop(['price_doc','sub_area','product_type'],axis=1)
        s1=s1.iloc[:,2:20]
        for fea in s1.columns:
                s1[fea].fillna(s1[fea].median(), inplace=True)
        for fea in s1.columns:
                s1[fea].fillna(0, inplace=True)
        s2=s2.drop(['price_doc','sub_area','product_type'],axis=1)
        s2=s2.iloc[:,2:20]
        for fea in s2.columns:
                s2[fea].fillna(s2[fea].median(), inplace=True)
        for fea in s2.columns:
                s2[fea].fillna(0, inplace=True)              
        s3=s3.drop(['price_doc','sub_area','product_type'],axis=1)
        s3=s3.iloc[:,2:20]
        for fea in s3.columns:
                s3[fea].fillna(s3[fea].median(), inplace=True)
        for fea in s3.columns:
                s3[fea].fillna(0, inplace=True)
        s4=s4.drop(['price_doc','sub_area','product_type'],axis=1)
        s4=s4.iloc[:,2:20]
        for fea in s4.columns:
                s4[fea].fillna(s4[fea].median(), inplace=True)
        for fea in s4.columns:
                s4[fea].fillna(0, inplace=True)
        
        sa=sa.drop(['price_doc','sub_area','product_type'],axis=1)
        sa=sa.iloc[:,2:20]
        for fea in sa.columns:
                sa[fea].fillna(sa[fea].median(), inplace=True)
        for fea in sa.columns:
                sa[fea].fillna(0, inplace=True)
                
                
        if self.FLAG1==1:
           X23 = pd.concat([s2,s3]) 
           y23 = np.concatenate([t2,t3],axis=0)
           random_estimator23=RandomForestRegressor(random_state=0, n_estimators=500)
           random_estimator23.fit(X23,y23)
           self.random_estimator2=random_estimator23
           self.random_estimator3=random_estimator23
           
           X23r=X23.full_sq
           X23r=X23r.reshape(len(X23r),1)
           y23r=y23.reshape(len(y23), 1)
           Linear_estimator23=LinearRegression()
           Linear_estimator23.fit(X23r, y23r)
           
           self.Linear_estimator2=Linear_estimator23
           self.Linear_estimator3=Linear_estimator23
           
        else:
            
            random_estimator2 = RandomForestRegressor(random_state=0, n_estimators=500)
            random_estimator2.fit(s2,t2)
            self.random_estimator2=random_estimator2
            
            Linear_estimator2 = LinearRegression()
            
            s2r=s2.full_sq
            s2r=s2r.reshape(len(s2r),1)
            t2r=t2
            t2r=t2r.reshape(len(t2r),1)
            Linear_estimator2.fit(s2r, t2r)
            self.Linear_estimator2=Linear_estimator2
            
            
            
            random_estimator3 = RandomForestRegressor(random_state=0, n_estimators=500)
            random_estimator3.fit(s3,t3)
            self.random_estimator3=random_estimator3
            
            Linear_estimator3 = LinearRegression()
            
            s3r=s3.full_sq
            s3r=s3r.reshape(len(s3r),1)
            t3r=t3
            t3r=t3r.reshape(len(t3r),1)
            Linear_estimator3.fit(s3r, t3r)
            self.Linear_estimator3=Linear_estimator3
            
            
            
           
        if self.FLAG2==1:
               self.TwoClassModel=twoclass
        else :
               
                    
                d3=np.ones(len(t3))
                d4=np.ones(len(t4))*-1
                          
                X = pd.concat([s3,s4])
                y = np.concatenate([d3,d4],axis=0)
                self.y=y
                self.X=X
                #print (X)
                #self.decision_clf = svm.SVC()
                TwoClassModel = RandomForestClassifier()
                TwoClassModel.fit(X, y)
                self.TwoClassModel=TwoClassModel
                
            
        Linear_estimator4 = LinearRegression()        
        s4r=s4.full_sq
        s4r=s4r.reshape(len(s4r),1)
        t4r=t4
        t4r=t4r.reshape(len(t4r),1)
        Linear_estimator4.fit(s4r, t4r)
        self.Linear_estimator4=Linear_estimator4             

        
    def predict_price(self,test_row):
        testrow1=test_row
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
        if dum > self.highcutoff1:
            #predicted_price=5
            predicted_price1= self.random_estimator3.predict(test_row)[0] 
            
            x1 = test_row.full_sq
            x1 = x1.reshape(len(x1),1)
            
            predicted_price2= self.Linear_estimator3.predict(x1)[0]
            
        
            predicted_price=0.2*predicted_price1+ 0.8*predicted_price2
            
            
        else:
            if flag==1:
                
                predicted_price1= self.random_estimator2.predict(test_row)[0] 
                x1 = test_row.full_sq
                x1 = x1.reshape(len(x1),1)
                predicted_price2= self.Linear_estimator2.predict(x1)[0]
                predicted_price=0.5*predicted_price1+ 0.5*predicted_price2
            
            else:
                if self.FLAG2==1:
                    testrow1=testrow1.drop(['sub_area', 'full_sq' ,'product_type'],axis=1)
                    testrow1=testrow1.iloc[:,2:20]
                    for fea in testrow1.columns:
                              testrow1[fea].fillna(testrow1[fea].median(),inplace=True)
                    for fea in testrow1.columns:
                              testrow1[fea].fillna(0,inplace=True)
                              
                    pclass=self.TwoClassModel.predict(testrow1)[0]
                
                else:
                       pclass=self.TwoClassModel.predict(test_row)[0]
                       
                if pclass==-1:
                       #predicted_price= 1.5
                       x1 = test_row.full_sq
                       x1 = x1.reshape(len(x1),1)
                       predicted_price= self.Linear_estimator4.predict(x1)[0]
                else:                        
                       predicted_price1= self.random_estimator3.predict(test_row)[0] 
                       x1 = test_row.full_sq
                       x1 = x1.reshape(len(x1),1)
                       predicted_price2= self.Linear_estimator3.predict(x1)[0]
                       predicted_price=1*predicted_price1+ 0*predicted_price2
                       
        if self.specialFlag==1:
            #if (dum >70) & (dum <600) & (predicted_price > 10):
            #     predicted_price=predicted_price + 3
            if (dum < 26)  :
                 predicted_price -= 2
        #predicted_price -= .1
        return predicted_price 

gc.collect()
###############################################################

class Birjulevo_Zapadnoe():
    #_data = pd.DataFrame
    def __init__(self, train_rows, twoclass):
        
        #### Flags
        self.specialFlag=0  ### this is for special adjustment in predict
        self.FLAG1=1  ### this is 1 for combined s2 s3
        self.FLAG2=1 ### this is 1 for global TwoClass classifier
        
        
        
        self.highcutoff = 150
        self.highcutoff1=150
        self.lowpricecutoff =4.2

        s1=train_rows[train_rows.full_sq > self.highcutoff];
        sa=train_rows[train_rows.full_sq < self.highcutoff];
        s2=sa[(sa.state.isnull()) & (sa.life_sq.isnull())]
        s3a = sa[~((sa.state.isnull()) & (sa.life_sq.isnull()))]
        s3=s3a[s3a.price_doc>self.lowpricecutoff]
        s4=s3a[s3a.price_doc <= self.lowpricecutoff]
        t1=s1.price_doc
        t2=s2.price_doc
        t3=s3.price_doc
        t4=s4.price_doc
        s1=s1.drop(['price_doc','sub_area','product_type'],axis=1)
        s1=s1.iloc[:,2:20]
        for fea in s1.columns:
                s1[fea].fillna(s1[fea].median(), inplace=True)
        for fea in s1.columns:
                s1[fea].fillna(0, inplace=True)
        s2=s2.drop(['price_doc','sub_area','product_type'],axis=1)
        s2=s2.iloc[:,2:20]
        for fea in s2.columns:
                s2[fea].fillna(s2[fea].median(), inplace=True)
        for fea in s2.columns:
                s2[fea].fillna(0, inplace=True)              
        s3=s3.drop(['price_doc','sub_area','product_type'],axis=1)
        s3=s3.iloc[:,2:20]
        for fea in s3.columns:
                s3[fea].fillna(s3[fea].median(), inplace=True)
        for fea in s3.columns:
                s3[fea].fillna(0, inplace=True)
        s4=s4.drop(['price_doc','sub_area','product_type'],axis=1)
        s4=s4.iloc[:,2:20]
        for fea in s4.columns:
                s4[fea].fillna(s4[fea].median(), inplace=True)
        for fea in s4.columns:
                s4[fea].fillna(0, inplace=True)
        
        sa=sa.drop(['price_doc','sub_area','product_type'],axis=1)
        sa=sa.iloc[:,2:20]
        for fea in sa.columns:
                sa[fea].fillna(sa[fea].median(), inplace=True)
        for fea in sa.columns:
                sa[fea].fillna(0, inplace=True)
                
                
        if self.FLAG1==1:
           X23 = pd.concat([s2,s3]) 
           y23 = np.concatenate([t2,t3],axis=0)
           random_estimator23=RandomForestRegressor(random_state=0, n_estimators=500)
           random_estimator23.fit(X23,y23)
           self.random_estimator2=random_estimator23
           self.random_estimator3=random_estimator23
           
           X23r=X23.full_sq
           X23r=X23r.reshape(len(X23r),1)
           y23r=y23.reshape(len(y23), 1)
           Linear_estimator23=LinearRegression()
           Linear_estimator23.fit(X23r, y23r)
           
           self.Linear_estimator2=Linear_estimator23
           self.Linear_estimator3=Linear_estimator23
           
        else:
            
            random_estimator2 = RandomForestRegressor(random_state=0, n_estimators=500)
            random_estimator2.fit(s2,t2)
            self.random_estimator2=random_estimator2
            
            Linear_estimator2 = LinearRegression()
            
            s2r=s2.full_sq
            s2r=s2r.reshape(len(s2r),1)
            t2r=t2
            t2r=t2r.reshape(len(t2r),1)
            Linear_estimator2.fit(s2r, t2r)
            self.Linear_estimator2=Linear_estimator2
            
            
            
            random_estimator3 = RandomForestRegressor(random_state=0, n_estimators=500)
            random_estimator3.fit(s3,t3)
            self.random_estimator3=random_estimator3
            
            Linear_estimator3 = LinearRegression()
            
            s3r=s3.full_sq
            s3r=s3r.reshape(len(s3r),1)
            t3r=t3
            t3r=t3r.reshape(len(t3r),1)
            Linear_estimator3.fit(s3r, t3r)
            self.Linear_estimator3=Linear_estimator3
            
            
            
           
        if self.FLAG2==1:
               self.TwoClassModel=twoclass
        else :
               
                    
                d3=np.ones(len(t3))
                d4=np.ones(len(t4))*-1
                          
                X = pd.concat([s3,s4])
                y = np.concatenate([d3,d4],axis=0)
                self.y=y
                self.X=X
                #print (X)
                #self.decision_clf = svm.SVC()
                TwoClassModel = RandomForestClassifier()
                TwoClassModel.fit(X, y)
                self.TwoClassModel=TwoClassModel
                
            
        Linear_estimator4 = LinearRegression()        
        s4r=s4.full_sq
        s4r=s4r.reshape(len(s4r),1)
        t4r=t4
        t4r=t4r.reshape(len(t4r),1)
        Linear_estimator4.fit(s4r, t4r)
        self.Linear_estimator4=Linear_estimator4             

        
    def predict_price(self,test_row):
        testrow1=test_row
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
        if dum > self.highcutoff1:
            #predicted_price=5
            predicted_price1= self.random_estimator3.predict(test_row)[0] 
            
            x1 = test_row.full_sq
            x1 = x1.reshape(len(x1),1)
            
            predicted_price2= self.Linear_estimator3.predict(x1)[0]
            
        
            predicted_price=0.2*predicted_price1+ 0.8*predicted_price2
            
            
        else:
            if flag==1:
                
                predicted_price1= self.random_estimator2.predict(test_row)[0] 
                x1 = test_row.full_sq
                x1 = x1.reshape(len(x1),1)
                predicted_price2= self.Linear_estimator2.predict(x1)[0]
                predicted_price=0.5*predicted_price1+ 0.5*predicted_price2
            
            else:
                if self.FLAG2==1:
                    testrow1=testrow1.drop(['sub_area', 'full_sq' ,'product_type'],axis=1)
                    testrow1=testrow1.iloc[:,2:20]
                    for fea in testrow1.columns:
                              testrow1[fea].fillna(testrow1[fea].median(),inplace=True)
                    for fea in testrow1.columns:
                              testrow1[fea].fillna(0,inplace=True)
                              
                    pclass=self.TwoClassModel.predict(testrow1)[0]
                
                else:
                       pclass=self.TwoClassModel.predict(test_row)[0]
                       
                if pclass==-1:
                       #predicted_price= 1.5
                       x1 = test_row.full_sq
                       x1 = x1.reshape(len(x1),1)
                       predicted_price= self.Linear_estimator4.predict(x1)[0]
                else:                        
                       predicted_price1= self.random_estimator3.predict(test_row)[0] 
                       x1 = test_row.full_sq
                       x1 = x1.reshape(len(x1),1)
                       predicted_price2= self.Linear_estimator3.predict(x1)[0]
                       predicted_price=1*predicted_price1+ 0*predicted_price2
                       
        if self.specialFlag==1:
            #if (dum >70) & (dum <600) & (predicted_price > 10):
            #     predicted_price=predicted_price + 3
            if (dum < 26)  :
                 predicted_price -= 2
        #predicted_price -= .1
        return predicted_price 

gc.collect()
###############################################################

class Donskoe():
    #_data = pd.DataFrame
    def __init__(self, train_rows, twoclass):
        
        #### Flags
        self.specialFlag=1  ### this is for special adjustment in predict
        self.FLAG1=0  ### this is 1 for combined s2 s3
        self.FLAG2=1 ### this is 1 for global TwoClass classifier
        
        
        
        self.highcutoff = 150
        self.highcutoff1=150
        self.lowpricecutoff =3

        s1=train_rows[train_rows.full_sq > self.highcutoff];
        sa=train_rows[train_rows.full_sq < self.highcutoff];
        s2=sa[(sa.state.isnull()) & (sa.life_sq.isnull())]
        s3a = sa[~((sa.state.isnull()) & (sa.life_sq.isnull()))]
        s3=s3a[s3a.price_doc>self.lowpricecutoff]
        s4=s3a[s3a.price_doc <= self.lowpricecutoff]
        t1=s1.price_doc
        t2=s2.price_doc
        t3=s3.price_doc
        t4=s4.price_doc
        s1=s1.drop(['price_doc','sub_area','product_type'],axis=1)
        s1=s1.iloc[:,2:20]
        for fea in s1.columns:
                s1[fea].fillna(s1[fea].median(), inplace=True)
        for fea in s1.columns:
                s1[fea].fillna(0, inplace=True)
        s2=s2.drop(['price_doc','sub_area','product_type'],axis=1)
        s2=s2.iloc[:,2:20]
        for fea in s2.columns:
                s2[fea].fillna(s2[fea].median(), inplace=True)
        for fea in s2.columns:
                s2[fea].fillna(0, inplace=True)              
        s3=s3.drop(['price_doc','sub_area','product_type'],axis=1)
        s3=s3.iloc[:,2:20]
        for fea in s3.columns:
                s3[fea].fillna(s3[fea].median(), inplace=True)
        for fea in s3.columns:
                s3[fea].fillna(0, inplace=True)
        s4=s4.drop(['price_doc','sub_area','product_type'],axis=1)
        s4=s4.iloc[:,2:20]
        for fea in s4.columns:
                s4[fea].fillna(s4[fea].median(), inplace=True)
        for fea in s4.columns:
                s4[fea].fillna(0, inplace=True)
        
        sa=sa.drop(['price_doc','sub_area','product_type'],axis=1)
        sa=sa.iloc[:,2:20]
        for fea in sa.columns:
                sa[fea].fillna(sa[fea].median(), inplace=True)
        for fea in sa.columns:
                sa[fea].fillna(0, inplace=True)
                
                
        if self.FLAG1==1:
           X23 = pd.concat([s2,s3]) 
           y23 = np.concatenate([t2,t3],axis=0)
           random_estimator23=RandomForestRegressor(random_state=0, n_estimators=500)
           random_estimator23.fit(X23,y23)
           self.random_estimator2=random_estimator23
           self.random_estimator3=random_estimator23
           
           X23r=X23.full_sq
           X23r=X23r.reshape(len(X23r),1)
           y23r=y23.reshape(len(y23), 1)
           Linear_estimator23=LinearRegression()
           Linear_estimator23.fit(X23r, y23r)
           
           self.Linear_estimator2=Linear_estimator23
           self.Linear_estimator3=Linear_estimator23
           
        else:
            
            random_estimator2 = RandomForestRegressor(random_state=0, n_estimators=500)
            random_estimator2.fit(s2,t2)
            self.random_estimator2=random_estimator2
            
            Linear_estimator2 = LinearRegression()
            
            s2r=s2.full_sq
            s2r=s2r.reshape(len(s2r),1)
            t2r=t2
            t2r=t2r.reshape(len(t2r),1)
            Linear_estimator2.fit(s2r, t2r)
            self.Linear_estimator2=Linear_estimator2
            
            
            
            random_estimator3 = RandomForestRegressor(random_state=0, n_estimators=500)
            random_estimator3.fit(s3,t3)
            self.random_estimator3=random_estimator3
            
            Linear_estimator3 = LinearRegression()
            
            s3r=s3.full_sq
            s3r=s3r.reshape(len(s3r),1)
            t3r=t3
            t3r=t3r.reshape(len(t3r),1)
            Linear_estimator3.fit(s3r, t3r)
            self.Linear_estimator3=Linear_estimator3
            
            
            
           
        if self.FLAG2==1:
               self.TwoClassModel=twoclass
        else :
               
                    
                d3=np.ones(len(t3))
                d4=np.ones(len(t4))*-1
                          
                X = pd.concat([s3,s4])
                y = np.concatenate([d3,d4],axis=0)
                self.y=y
                self.X=X
                #print (X)
                #self.decision_clf = svm.SVC()
                TwoClassModel = RandomForestClassifier()
                TwoClassModel.fit(X, y)
                self.TwoClassModel=TwoClassModel
                
            
        Linear_estimator4 = LinearRegression()        
        s4r=s4.full_sq
        s4r=s4r.reshape(len(s4r),1)
        t4r=t4
        t4r=t4r.reshape(len(t4r),1)
        Linear_estimator4.fit(s4r, t4r)
        self.Linear_estimator4=Linear_estimator4             

        
    def predict_price(self,test_row):
        testrow1=test_row
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
        if dum > self.highcutoff1:
            #predicted_price=5
            predicted_price1= self.random_estimator3.predict(test_row)[0] 
            
            x1 = test_row.full_sq
            x1 = x1.reshape(len(x1),1)
            
            predicted_price2= self.Linear_estimator3.predict(x1)[0]
            
        
            predicted_price=0.1*predicted_price1+ 0.9*predicted_price2
            
            
        else:
            if flag==1:
                
                predicted_price1= self.random_estimator2.predict(test_row)[0] 
                x1 = test_row.full_sq
                x1 = x1.reshape(len(x1),1)
                predicted_price2= self.Linear_estimator2.predict(x1)[0]
                predicted_price=0.75*predicted_price1+ 0.25*predicted_price2
                predicted_price -= 10
            else:
                if self.FLAG2==1:
                    testrow1=testrow1.drop(['sub_area', 'full_sq' ,'product_type'],axis=1)
                    testrow1=testrow1.iloc[:,2:20]
                    for fea in testrow1.columns:
                              testrow1[fea].fillna(testrow1[fea].median(),inplace=True)
                    for fea in testrow1.columns:
                              testrow1[fea].fillna(0,inplace=True)
                              
                    pclass=self.TwoClassModel.predict(testrow1)[0]
                
                else:
                       pclass=self.TwoClassModel.predict(test_row)[0]
                       
                if pclass==-1:
                       #predicted_price= 1.5
                       x1 = test_row.full_sq
                       x1 = x1.reshape(len(x1),1)
                       predicted_price= self.Linear_estimator4.predict(x1)[0]
                else:                        
                       predicted_price1= self.random_estimator3.predict(test_row)[0] 
                       x1 = test_row.full_sq
                       x1 = x1.reshape(len(x1),1)
                       predicted_price2= self.Linear_estimator3.predict(x1)[0]
                       predicted_price=.9*predicted_price1+ 0.1*predicted_price2
                       
        if self.specialFlag==1:
            #if (dum >70) & (dum <600) & (predicted_price > 10):
            #     predicted_price=predicted_price + 3
            if (dum < 30)  :
                 predicted_price -= 5
            if (dum > 100):
                 predicted_price += 8
        #predicted_price -= .1
        return predicted_price 

gc.collect()
###############################################################





###### All small class will go here

class Small_Class_Model():
    #_data = pd.DataFrame
    def __init__(self, train_rows, twoclass):
        
        #### Flags
        self.specialFlag=0 ### this is for special adjustment in predict
        self.FLAG1=1 ### this is 1 for combined s2 s3
        self.FLAG2=1 ### this is 1 for global TwoClass classifier
        
        
        
        self.highcutoff = 150
        self.highcutoff1=150
        self.lowpricecutoff = 2.2

        s1=train_rows[train_rows.full_sq > self.highcutoff];
        sa=train_rows[train_rows.full_sq < self.highcutoff];
        s2=sa[(sa.state.isnull()) & (sa.life_sq.isnull())]
        s3a = sa[~((sa.state.isnull()) & (sa.life_sq.isnull()))]
        s3=s3a[s3a.price_doc>self.lowpricecutoff]
        s4=s3a[s3a.price_doc <= self.lowpricecutoff]
        t1=s1.price_doc
        t2=s2.price_doc
        t3=s3.price_doc
        t4=s4.price_doc
        s1=s1.drop(['price_doc','sub_area','product_type'],axis=1)
        s1=s1.iloc[:,2:20]
        for fea in s1.columns:
                s1[fea].fillna(s1[fea].median(), inplace=True)
        for fea in s1.columns:
                s1[fea].fillna(0, inplace=True)
        s2=s2.drop(['price_doc','sub_area','product_type'],axis=1)
        s2=s2.iloc[:,2:20]
        for fea in s2.columns:
                s2[fea].fillna(s2[fea].median(), inplace=True)
        for fea in s2.columns:
                s2[fea].fillna(0, inplace=True)              
        s3=s3.drop(['price_doc','sub_area','product_type'],axis=1)
        s3=s3.iloc[:,2:20]
        for fea in s3.columns:
                s3[fea].fillna(s3[fea].median(), inplace=True)
        for fea in s3.columns:
                s3[fea].fillna(0, inplace=True)
        s4=s4.drop(['price_doc','sub_area','product_type'],axis=1)
        s4=s4.iloc[:,2:20]
        for fea in s4.columns:
                s4[fea].fillna(s4[fea].median(), inplace=True)
        for fea in s4.columns:
                s4[fea].fillna(0, inplace=True)
        
        sa=sa.drop(['price_doc','sub_area','product_type'],axis=1)
        sa=sa.iloc[:,2:20]
        for fea in sa.columns:
                sa[fea].fillna(sa[fea].median(), inplace=True)
        for fea in sa.columns:
                sa[fea].fillna(0, inplace=True)
                
                
        if self.FLAG1==1:
           X23 = pd.concat([s2,s3]) 
           y23 = np.concatenate([t2,t3],axis=0)
           random_estimator23=RandomForestRegressor(random_state=0, n_estimators=500)
           random_estimator23.fit(X23,y23)
           self.random_estimator2=random_estimator23
           self.random_estimator3=random_estimator23
           
           X23r=X23.full_sq
           X23r=X23r.reshape(len(X23r),1)
           y23r=y23.reshape(len(y23), 1)
           Linear_estimator23=LinearRegression()
           Linear_estimator23.fit(X23r, y23r)
           
           self.Linear_estimator2=Linear_estimator23
           self.Linear_estimator3=Linear_estimator23
           
        else:
            
            random_estimator2 = RandomForestRegressor(random_state=0, n_estimators=500)
            random_estimator2.fit(s2,t2)
            self.random_estimator2=random_estimator2
            
            Linear_estimator2 = LinearRegression()
            
            s2r=s2.full_sq
            s2r=s2r.reshape(len(s2r),1)
            t2r=t2
            t2r=t2r.reshape(len(t2r),1)
            Linear_estimator2.fit(s2r, t2r)
            self.Linear_estimator2=Linear_estimator2
            
            
            
            random_estimator3 = RandomForestRegressor(random_state=0, n_estimators=500)
            random_estimator3.fit(s3,t3)
            self.random_estimator3=random_estimator3
            
            Linear_estimator3 = LinearRegression()
            
            s3r=s3.full_sq
            s3r=s3r.reshape(len(s3r),1)
            t3r=t3
            t3r=t3r.reshape(len(t3r),1)
            Linear_estimator3.fit(s3r, t3r)
            self.Linear_estimator3=Linear_estimator3
            
            
            
           
        if self.FLAG2==1:
               self.TwoClassModel=twoclass
        else :
               
                    
                d3=np.ones(len(t3))
                d4=np.ones(len(t4))*-1
                          
                X = pd.concat([s3,s4])
                y = np.concatenate([d3,d4],axis=0)
                self.y=y
                self.X=X
                #print (X)
                #self.decision_clf = svm.SVC()
                TwoClassModel = RandomForestClassifier()
                TwoClassModel.fit(X, y)
                self.TwoClassModel=TwoClassModel
                
            
        Linear_estimator4 = LinearRegression()        
        s4r=s4.full_sq
        s4r=s4r.reshape(len(s4r),1)
        t4r=t4
        t4r=t4r.reshape(len(t4r),1)
        Linear_estimator4.fit(s4r, t4r)
        self.Linear_estimator4=Linear_estimator4             

        
    def predict_price(self,test_row):
        testrow1=test_row
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
        if dum > self.highcutoff1:
            #predicted_price=5
            predicted_price1= self.random_estimator3.predict(test_row)[0] 
            
            x1 = test_row.full_sq
            x1 = x1.reshape(len(x1),1)
            
            predicted_price2= self.Linear_estimator3.predict(x1)[0]
            
        
            predicted_price=0.2*predicted_price1+ 0.8*predicted_price2
            
            
        else:
            if flag==1:
                
                predicted_price1= self.random_estimator2.predict(test_row)[0] 
                x1 = test_row.full_sq
                x1 = x1.reshape(len(x1),1)
                predicted_price2= self.Linear_estimator2.predict(x1)[0]
                predicted_price=0.4*predicted_price1+ 0.6*predicted_price2
            
            else:
                if self.FLAG2==1:
                    testrow1=testrow1.drop(['sub_area', 'full_sq' ,'product_type'],axis=1)
                    testrow1=testrow1.iloc[:,2:20]
                    for fea in testrow1.columns:
                              testrow1[fea].fillna(testrow1[fea].median(),inplace=True)
                    for fea in testrow1.columns:
                              testrow1[fea].fillna(0,inplace=True)
                              
                    pclass=self.TwoClassModel.predict(testrow1)[0]
                
                else:
                       pclass=self.TwoClassModel.predict(test_row)[0]
                       
                if pclass==-1:
                       #predicted_price= 1.5
                       x1 = test_row.full_sq
                       x1 = x1.reshape(len(x1),1)
                       predicted_price= self.Linear_estimator4.predict(x1)[0]
                else:                        
                       predicted_price1= self.random_estimator3.predict(test_row)[0] 
                       x1 = test_row.full_sq
                       x1 = x1.reshape(len(x1),1)
                       predicted_price2= self.Linear_estimator3.predict(x1)[0]
                       predicted_price=0.5*predicted_price1+ 0.5*predicted_price2
                       
        if self.specialFlag==1:
            if dum < 65 :
                 predicted_price=predicted_price-2
      
        return predicted_price


gc.collect()

#############################################################
#############################################################
##############################################################

small_area_train_selectedRows=(train_df.sub_area=='Poselenie Mihajlovo-Jarcevskoe') | \
             (train_df.sub_area=='Molzhaninovskoe') | \
             (train_df.sub_area=='Poselenie Kievskij') | \
             (train_df.sub_area=='Poselenie Klenovskoe') | \
             (train_df.sub_area=='Poselenie Voronovskoe') | \
             (train_df.sub_area=='Vostochnoe') | \
             (train_df.sub_area=='Poselenie Shhapovskoe') | \
             (train_df.sub_area=='Poselenie Marushkinskoe') #137
             
             
             
#small_area_test_selectedRows=(test_df.sub_area=='Poselenie Mihajlovo-Jarcevskoe') | \
#             (test_df.sub_area=='Molzhaninovskoe') | \
#             (test_df.sub_area=='Poselenie Kievskij') | \
#             (test_df.sub_area=='Poselenie Klenovskoe') | \
#             (test_df.sub_area=='Poselenie Voronovskoe') | \
#             (test_df.sub_area=='Vostochnoe') | \
#             (test_df.sub_area=='Poselenie Shhapovskoe') | \
#             (test_df.sub_area=='Poselenie Marushkinskoe') #137


model_all=MasterModel_ALL(train_df, TwoClassModel)
small_models=Small_Class_Model(train_df[small_area_train_selectedRows], TwoClassModel)



#############################################################################3
class Poselenie_Pervomajskoe():
    #_data = pd.DataFrame
    def __init__(self, train_rows, twoclass):
        
        #### Flags
        self.specialFlag=0 ### this is for special adjustment in predict
        self.FLAG1=1 ### this is 1 for combined s2 s3
        self.FLAG2=1 ### this is 1 for global TwoClass classifier
        
        
        
        self.highcutoff = 150
        self.highcutoff1=150
        self.lowpricecutoff = 2.2

        s1=train_rows[train_rows.full_sq > self.highcutoff];
        sa=train_rows[train_rows.full_sq < self.highcutoff];
        s2=sa[(sa.state.isnull()) & (sa.life_sq.isnull())]
        s3a = sa[~((sa.state.isnull()) & (sa.life_sq.isnull()))]
        s3=s3a[s3a.price_doc>self.lowpricecutoff]
        s4=s3a[s3a.price_doc <= self.lowpricecutoff]
        t1=s1.price_doc
        t2=s2.price_doc
        t3=s3.price_doc
        t4=s4.price_doc
        s1=s1.drop(['price_doc','sub_area','product_type'],axis=1)
        s1=s1.iloc[:,2:20]
        for fea in s1.columns:
                s1[fea].fillna(s1[fea].median(), inplace=True)
        for fea in s1.columns:
                s1[fea].fillna(0, inplace=True)
        s2=s2.drop(['price_doc','sub_area','product_type'],axis=1)
        s2=s2.iloc[:,2:20]
        for fea in s2.columns:
                s2[fea].fillna(s2[fea].median(), inplace=True)
        for fea in s2.columns:
                s2[fea].fillna(0, inplace=True)              
        s3=s3.drop(['price_doc','sub_area','product_type'],axis=1)
        s3=s3.iloc[:,2:20]
        for fea in s3.columns:
                s3[fea].fillna(s3[fea].median(), inplace=True)
        for fea in s3.columns:
                s3[fea].fillna(0, inplace=True)
        s4=s4.drop(['price_doc','sub_area','product_type'],axis=1)
        s4=s4.iloc[:,2:20]
        for fea in s4.columns:
                s4[fea].fillna(s4[fea].median(), inplace=True)
        for fea in s4.columns:
                s4[fea].fillna(0, inplace=True)
        
        sa=sa.drop(['price_doc','sub_area','product_type'],axis=1)
        sa=sa.iloc[:,2:20]
        for fea in sa.columns:
                sa[fea].fillna(sa[fea].median(), inplace=True)
        for fea in sa.columns:
                sa[fea].fillna(0, inplace=True)
                
                
        if self.FLAG1==1:
           X23 = pd.concat([s2,s3]) 
           y23 = np.concatenate([t2,t3],axis=0)
           random_estimator23=RandomForestRegressor(random_state=0, n_estimators=500)
           random_estimator23.fit(X23,y23)
           self.random_estimator2=random_estimator23
           self.random_estimator3=random_estimator23
           
           X23r=X23.full_sq
           X23r=X23r.reshape(len(X23r),1)
           y23r=y23.reshape(len(y23), 1)
           Linear_estimator23=LinearRegression()
           Linear_estimator23.fit(X23r, y23r)
           
           self.Linear_estimator2=Linear_estimator23
           self.Linear_estimator3=Linear_estimator23
           
        else:
            
            random_estimator2 = RandomForestRegressor(random_state=0, n_estimators=500)
            random_estimator2.fit(s2,t2)
            self.random_estimator2=random_estimator2
            
            Linear_estimator2 = LinearRegression()
            
            s2r=s2.full_sq
            s2r=s2r.reshape(len(s2r),1)
            t2r=t2
            t2r=t2r.reshape(len(t2r),1)
            Linear_estimator2.fit(s2r, t2r)
            self.Linear_estimator2=Linear_estimator2
            
            
            
            random_estimator3 = RandomForestRegressor(random_state=0, n_estimators=500)
            random_estimator3.fit(s3,t3)
            self.random_estimator3=random_estimator3
            
            Linear_estimator3 = LinearRegression()
            
            s3r=s3.full_sq
            s3r=s3r.reshape(len(s3r),1)
            t3r=t3
            t3r=t3r.reshape(len(t3r),1)
            Linear_estimator3.fit(s3r, t3r)
            self.Linear_estimator3=Linear_estimator3
            
            
            
           
        if self.FLAG2==1:
               self.TwoClassModel=twoclass
        else :
               
                    
                d3=np.ones(len(t3))
                d4=np.ones(len(t4))*-1
                          
                X = pd.concat([s3,s4])
                y = np.concatenate([d3,d4],axis=0)
                self.y=y
                self.X=X
                #print (X)
                #self.decision_clf = svm.SVC()
                TwoClassModel = RandomForestClassifier()
                TwoClassModel.fit(X, y)
                self.TwoClassModel=TwoClassModel
                
        
        random_estimator4 = RandomForestRegressor(random_state=0, n_estimators=500)
        random_estimator4.fit(s4,t4)
        self.random_estimator4=random_estimator4
        
        Linear_estimator4 = LinearRegression()        
        s4r=s4.full_sq
        s4r=s4r.reshape(len(s4r),1)
        t4r=t4
        t4r=t4r.reshape(len(t4r),1)
        Linear_estimator4.fit(s4r, t4r)
        self.Linear_estimator4=Linear_estimator4             

        
    def predict_price(self,test_row):
        testrow1=test_row
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
        if dum > self.highcutoff1:
            #predicted_price=5
            predicted_price1= self.random_estimator3.predict(test_row)[0] 
            
            x1 = test_row.full_sq
            x1 = x1.reshape(len(x1),1)
            
            predicted_price2= self.Linear_estimator3.predict(x1)[0]
            
        
            predicted_price=0.2*predicted_price1+ 0.8*predicted_price2
            
            
        else:
            if flag==1:
                
                predicted_price1= self.random_estimator2.predict(test_row)[0] 
                x1 = test_row.full_sq
                x1 = x1.reshape(len(x1),1)
                predicted_price2= self.Linear_estimator2.predict(x1)[0]
                predicted_price=0.2*predicted_price1+ 0.8*predicted_price2
            
            else:
                if self.FLAG2==1:
                    testrow1=testrow1.drop(['sub_area', 'full_sq' ,'product_type'],axis=1)
                    testrow1=testrow1.iloc[:,2:20]
                    for fea in testrow1.columns:
                              testrow1[fea].fillna(testrow1[fea].median(),inplace=True)
                    for fea in testrow1.columns:
                              testrow1[fea].fillna(0,inplace=True)
                              
                    pclass=self.TwoClassModel.predict(testrow1)[0]
                
                else:
                       pclass=self.TwoClassModel.predict(test_row)[0]
                       
                if pclass==-2:
                       #predicted_price= 1.5
                       predicted_price1= self.random_estimator4.predict(test_row)[0]
                       x1 = test_row.full_sq
                       x1 = x1.reshape(len(x1),1)
                       predicted_price2= self.Linear_estimator4.predict(x1)[0]
                       predicted_price=0.9*predicted_price1+ 0.1*predicted_price2
                else:                        
                       predicted_price1= self.random_estimator3.predict(test_row)[0] 
                       x1 = test_row.full_sq
                       x1 = x1.reshape(len(x1),1)
                       predicted_price2= self.Linear_estimator3.predict(x1)[0]
                       predicted_price=0.2*predicted_price1+ 0.8*predicted_price2
                       
        if self.specialFlag==1:
            if dum < 65 :
                 predicted_price=predicted_price-2
      
        return predicted_price


######################################################################
class Poselenie_Kokoshkino():
    #_data = pd.DataFrame
    def __init__(self, train_rows, twoclass):
        
        #### Flags
        self.specialFlag=0 ### this is for special adjustment in predict
        self.FLAG1=1 ### this is 1 for combined s2 s3
        self.FLAG2=1 ### this is 1 for global TwoClass classifier
        
        
        
        self.highcutoff = 150
        self.highcutoff1=150
        self.lowpricecutoff = 2.2

        s1=train_rows[train_rows.full_sq > self.highcutoff];
        sa=train_rows[train_rows.full_sq < self.highcutoff];
        s2=sa[(sa.state.isnull()) & (sa.life_sq.isnull())]
        s3a = sa[~((sa.state.isnull()) & (sa.life_sq.isnull()))]
        s3=s3a[s3a.price_doc>self.lowpricecutoff]
        s4=s3a[s3a.price_doc <= self.lowpricecutoff]
        t1=s1.price_doc
        t2=s2.price_doc
        t3=s3.price_doc
        t4=s4.price_doc
        s1=s1.drop(['price_doc','sub_area','product_type'],axis=1)
        s1=s1.iloc[:,2:20]
        for fea in s1.columns:
                s1[fea].fillna(s1[fea].median(), inplace=True)
        for fea in s1.columns:
                s1[fea].fillna(0, inplace=True)
        s2=s2.drop(['price_doc','sub_area','product_type'],axis=1)
        s2=s2.iloc[:,2:20]
        for fea in s2.columns:
                s2[fea].fillna(s2[fea].median(), inplace=True)
        for fea in s2.columns:
                s2[fea].fillna(0, inplace=True)              
        s3=s3.drop(['price_doc','sub_area','product_type'],axis=1)
        s3=s3.iloc[:,2:20]
        for fea in s3.columns:
                s3[fea].fillna(s3[fea].median(), inplace=True)
        for fea in s3.columns:
                s3[fea].fillna(0, inplace=True)
        s4=s4.drop(['price_doc','sub_area','product_type'],axis=1)
        s4=s4.iloc[:,2:20]
        for fea in s4.columns:
                s4[fea].fillna(s4[fea].median(), inplace=True)
        for fea in s4.columns:
                s4[fea].fillna(0, inplace=True)
        
        sa=sa.drop(['price_doc','sub_area','product_type'],axis=1)
        sa=sa.iloc[:,2:20]
        for fea in sa.columns:
                sa[fea].fillna(sa[fea].median(), inplace=True)
        for fea in sa.columns:
                sa[fea].fillna(0, inplace=True)
                
                
        if self.FLAG1==1:
           X23 = pd.concat([s2,s3]) 
           y23 = np.concatenate([t2,t3],axis=0)
           random_estimator23=RandomForestRegressor(random_state=0, n_estimators=500)
           random_estimator23.fit(X23,y23)
           self.random_estimator2=random_estimator23
           self.random_estimator3=random_estimator23
           
           X23r=X23.full_sq
           X23r=X23r.reshape(len(X23r),1)
           y23r=y23.reshape(len(y23), 1)
           Linear_estimator23=LinearRegression()
           Linear_estimator23.fit(X23r, y23r)
           
           self.Linear_estimator2=Linear_estimator23
           self.Linear_estimator3=Linear_estimator23
           
        else:
            
            random_estimator2 = RandomForestRegressor(random_state=0, n_estimators=500)
            random_estimator2.fit(s2,t2)
            self.random_estimator2=random_estimator2
            
            Linear_estimator2 = LinearRegression()
            
            s2r=s2.full_sq
            s2r=s2r.reshape(len(s2r),1)
            t2r=t2
            t2r=t2r.reshape(len(t2r),1)
            Linear_estimator2.fit(s2r, t2r)
            self.Linear_estimator2=Linear_estimator2
            
            
            
            random_estimator3 = RandomForestRegressor(random_state=0, n_estimators=500)
            random_estimator3.fit(s3,t3)
            self.random_estimator3=random_estimator3
            
            Linear_estimator3 = LinearRegression()
            
            s3r=s3.full_sq
            s3r=s3r.reshape(len(s3r),1)
            t3r=t3
            t3r=t3r.reshape(len(t3r),1)
            Linear_estimator3.fit(s3r, t3r)
            self.Linear_estimator3=Linear_estimator3
            
            
            
           
        if self.FLAG2==1:
               self.TwoClassModel=twoclass
        else :
               
                    
                d3=np.ones(len(t3))
                d4=np.ones(len(t4))*-1
                          
                X = pd.concat([s3,s4])
                y = np.concatenate([d3,d4],axis=0)
                self.y=y
                self.X=X
                #print (X)
                #self.decision_clf = svm.SVC()
                TwoClassModel = RandomForestClassifier()
                TwoClassModel.fit(X, y)
                self.TwoClassModel=TwoClassModel
                
        
        random_estimator4 = RandomForestRegressor(random_state=0, n_estimators=500)
        random_estimator4.fit(s4,t4)
        self.random_estimator4=random_estimator4
        
        Linear_estimator4 = LinearRegression()        
        s4r=s4.full_sq
        s4r=s4r.reshape(len(s4r),1)
        t4r=t4
        t4r=t4r.reshape(len(t4r),1)
        Linear_estimator4.fit(s4r, t4r)
        self.Linear_estimator4=Linear_estimator4             

        
    def predict_price(self,test_row):
        testrow1=test_row
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
        if dum > self.highcutoff1:
            #predicted_price=5
            predicted_price1= self.random_estimator3.predict(test_row)[0] 
            
            x1 = test_row.full_sq
            x1 = x1.reshape(len(x1),1)
            
            predicted_price2= self.Linear_estimator3.predict(x1)[0]
            
        
            predicted_price=0.2*predicted_price1+ 0.8*predicted_price2
            
            
        else:
            if flag==1:
                
                predicted_price1= self.random_estimator2.predict(test_row)[0] 
                x1 = test_row.full_sq
                x1 = x1.reshape(len(x1),1)
                predicted_price2= self.Linear_estimator2.predict(x1)[0]
                predicted_price=0.2*predicted_price1+ 0.8*predicted_price2
            
            else:
                if self.FLAG2==1:
                    testrow1=testrow1.drop(['sub_area', 'full_sq' ,'product_type'],axis=1)
                    testrow1=testrow1.iloc[:,2:20]
                    for fea in testrow1.columns:
                              testrow1[fea].fillna(testrow1[fea].median(),inplace=True)
                    for fea in testrow1.columns:
                              testrow1[fea].fillna(0,inplace=True)
                              
                    pclass=self.TwoClassModel.predict(testrow1)[0]
                
                else:
                       pclass=self.TwoClassModel.predict(test_row)[0]
                       
                if pclass==-2:
                       #predicted_price= 1.5
                       predicted_price1= self.random_estimator4.predict(test_row)[0]
                       x1 = test_row.full_sq
                       x1 = x1.reshape(len(x1),1)
                       predicted_price2= self.Linear_estimator4.predict(x1)[0]
                       predicted_price=0.9*predicted_price1+ 0.1*predicted_price2
                else:                        
                       predicted_price1= self.random_estimator3.predict(test_row)[0] 
                       x1 = test_row.full_sq
                       x1 = x1.reshape(len(x1),1)
                       predicted_price2= self.Linear_estimator3.predict(x1)[0]
                       predicted_price=0.2*predicted_price1+ 0.8*predicted_price2
                       
        if self.specialFlag==1:
            if dum < 65 :
                 predicted_price=predicted_price-2
      
        return predicted_price

gc.collect()
#########################################################
class Metrogorodok():
    #_data = pd.DataFrame
    def __init__(self, train_rows, twoclass):
        
        #### Flags
        self.specialFlag=0 ### this is for special adjustment in predict
        self.FLAG1=1 ### this is 1 for combined s2 s3
        self.FLAG2=1 ### this is 1 for global TwoClass classifier
        
        
        
        self.highcutoff = 150
        self.highcutoff1=150
        self.lowpricecutoff = 2.2

        s1=train_rows[train_rows.full_sq > self.highcutoff];
        sa=train_rows[train_rows.full_sq < self.highcutoff];
        s2=sa[(sa.state.isnull()) & (sa.life_sq.isnull())]
        s3a = sa[~((sa.state.isnull()) & (sa.life_sq.isnull()))]
        s3=s3a[s3a.price_doc>self.lowpricecutoff]
        s4=s3a[s3a.price_doc <= self.lowpricecutoff]
        t1=s1.price_doc
        t2=s2.price_doc
        t3=s3.price_doc
        t4=s4.price_doc
        s1=s1.drop(['price_doc','sub_area','product_type'],axis=1)
        s1=s1.iloc[:,2:20]
        for fea in s1.columns:
                s1[fea].fillna(s1[fea].median(), inplace=True)
        for fea in s1.columns:
                s1[fea].fillna(0, inplace=True)
        s2=s2.drop(['price_doc','sub_area','product_type'],axis=1)
        s2=s2.iloc[:,2:20]
        for fea in s2.columns:
                s2[fea].fillna(s2[fea].median(), inplace=True)
        for fea in s2.columns:
                s2[fea].fillna(0, inplace=True)              
        s3=s3.drop(['price_doc','sub_area','product_type'],axis=1)
        s3=s3.iloc[:,2:20]
        for fea in s3.columns:
                s3[fea].fillna(s3[fea].median(), inplace=True)
        for fea in s3.columns:
                s3[fea].fillna(0, inplace=True)
        s4=s4.drop(['price_doc','sub_area','product_type'],axis=1)
        s4=s4.iloc[:,2:20]
        for fea in s4.columns:
                s4[fea].fillna(s4[fea].median(), inplace=True)
        for fea in s4.columns:
                s4[fea].fillna(0, inplace=True)
        
        sa=sa.drop(['price_doc','sub_area','product_type'],axis=1)
        sa=sa.iloc[:,2:20]
        for fea in sa.columns:
                sa[fea].fillna(sa[fea].median(), inplace=True)
        for fea in sa.columns:
                sa[fea].fillna(0, inplace=True)
                
                
        if self.FLAG1==1:
           X23 = pd.concat([s2,s3]) 
           y23 = np.concatenate([t2,t3],axis=0)
           random_estimator23=RandomForestRegressor(random_state=0, n_estimators=500)
           random_estimator23.fit(X23,y23)
           self.random_estimator2=random_estimator23
           self.random_estimator3=random_estimator23
           
           X23r=X23.full_sq
           X23r=X23r.reshape(len(X23r),1)
           y23r=y23.reshape(len(y23), 1)
           Linear_estimator23=LinearRegression()
           Linear_estimator23.fit(X23r, y23r)
           
           self.Linear_estimator2=Linear_estimator23
           self.Linear_estimator3=Linear_estimator23
           
        else:
            
            random_estimator2 = RandomForestRegressor(random_state=0, n_estimators=500)
            random_estimator2.fit(s2,t2)
            self.random_estimator2=random_estimator2
            
            Linear_estimator2 = LinearRegression()
            
            s2r=s2.full_sq
            s2r=s2r.reshape(len(s2r),1)
            t2r=t2
            t2r=t2r.reshape(len(t2r),1)
            Linear_estimator2.fit(s2r, t2r)
            self.Linear_estimator2=Linear_estimator2
            
            
            
            random_estimator3 = RandomForestRegressor(random_state=0, n_estimators=500)
            random_estimator3.fit(s3,t3)
            self.random_estimator3=random_estimator3
            
            Linear_estimator3 = LinearRegression()
            
            s3r=s3.full_sq
            s3r=s3r.reshape(len(s3r),1)
            t3r=t3
            t3r=t3r.reshape(len(t3r),1)
            Linear_estimator3.fit(s3r, t3r)
            self.Linear_estimator3=Linear_estimator3
            
            
            
           
        if self.FLAG2==1:
               self.TwoClassModel=twoclass
        else :
               
                    
                d3=np.ones(len(t3))
                d4=np.ones(len(t4))*-1
                          
                X = pd.concat([s3,s4])
                y = np.concatenate([d3,d4],axis=0)
                self.y=y
                self.X=X
                #print (X)
                #self.decision_clf = svm.SVC()
                TwoClassModel = RandomForestClassifier()
                TwoClassModel.fit(X, y)
                self.TwoClassModel=TwoClassModel
                
        
        random_estimator4 = RandomForestRegressor(random_state=0, n_estimators=500)
        random_estimator4.fit(s4,t4)
        self.random_estimator4=random_estimator4
        
        Linear_estimator4 = LinearRegression()        
        s4r=s4.full_sq
        s4r=s4r.reshape(len(s4r),1)
        t4r=t4
        t4r=t4r.reshape(len(t4r),1)
        Linear_estimator4.fit(s4r, t4r)
        self.Linear_estimator4=Linear_estimator4             

        
    def predict_price(self,test_row):
        testrow1=test_row
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
        if dum > self.highcutoff1:
            #predicted_price=5
            predicted_price1= self.random_estimator3.predict(test_row)[0] 
            
            x1 = test_row.full_sq
            x1 = x1.reshape(len(x1),1)
            
            predicted_price2= self.Linear_estimator3.predict(x1)[0]
            
        
            predicted_price=0.2*predicted_price1+ 0.8*predicted_price2
            
            
        else:
            if flag==1:
                
                predicted_price1= self.random_estimator2.predict(test_row)[0] 
                x1 = test_row.full_sq
                x1 = x1.reshape(len(x1),1)
                predicted_price2= self.Linear_estimator2.predict(x1)[0]
                predicted_price=0.2*predicted_price1+ 0.8*predicted_price2
            
            else:
                if self.FLAG2==1:
                    testrow1=testrow1.drop(['sub_area', 'full_sq' ,'product_type'],axis=1)
                    testrow1=testrow1.iloc[:,2:20]
                    for fea in testrow1.columns:
                              testrow1[fea].fillna(testrow1[fea].median(),inplace=True)
                    for fea in testrow1.columns:
                              testrow1[fea].fillna(0,inplace=True)
                              
                    pclass=self.TwoClassModel.predict(testrow1)[0]
                
                else:
                       pclass=self.TwoClassModel.predict(test_row)[0]
                       
                if pclass==-2:
                       #predicted_price= 1.5
                       predicted_price1= self.random_estimator4.predict(test_row)[0]
                       x1 = test_row.full_sq
                       x1 = x1.reshape(len(x1),1)
                       predicted_price2= self.Linear_estimator4.predict(x1)[0]
                       predicted_price=0.9*predicted_price1+ 0.1*predicted_price2
                else:                        
                       predicted_price1= self.random_estimator3.predict(test_row)[0] 
                       x1 = test_row.full_sq
                       x1 = x1.reshape(len(x1),1)
                       predicted_price2= self.Linear_estimator3.predict(x1)[0]
                       predicted_price=0.2*predicted_price1+ 0.8*predicted_price2
                       
        if self.specialFlag==1:
            if dum < 65 :
                 predicted_price=predicted_price-2
      
        return predicted_price

############################################
class LinearModel():
    #_data = pd.DataFrame
    def __init__(self, train_rows, twoclass):
        
        #### Flags
        self.specialFlag=0 ### this is for special adjustment in predict
        self.FLAG1=1 ### this is 1 for combined s2 s3
        self.FLAG2=1 ### this is 1 for global TwoClass classifier
        
        
        
        self.highcutoff = 150
        self.highcutoff1=150
        self.lowpricecutoff = 2.2

        s1=train_rows[train_rows.full_sq > self.highcutoff];
        sa=train_rows[train_rows.full_sq < self.highcutoff];
        s2=sa[(sa.state.isnull()) & (sa.life_sq.isnull())]
        s3a = sa[~((sa.state.isnull()) & (sa.life_sq.isnull()))]
        s3=s3a[s3a.price_doc>self.lowpricecutoff]
        s4=s3a[s3a.price_doc <= self.lowpricecutoff]
        t1=s1.price_doc
        t2=s2.price_doc
        t3=s3.price_doc
        t4=s4.price_doc
        s1=s1.drop(['price_doc','sub_area','product_type'],axis=1)
        s1=s1.iloc[:,2:20]
        for fea in s1.columns:
                s1[fea].fillna(s1[fea].median(), inplace=True)
        for fea in s1.columns:
                s1[fea].fillna(0, inplace=True)
        s2=s2.drop(['price_doc','sub_area','product_type'],axis=1)
        s2=s2.iloc[:,2:20]
        for fea in s2.columns:
                s2[fea].fillna(s2[fea].median(), inplace=True)
        for fea in s2.columns:
                s2[fea].fillna(0, inplace=True)              
        s3=s3.drop(['price_doc','sub_area','product_type'],axis=1)
        s3=s3.iloc[:,2:20]
        for fea in s3.columns:
                s3[fea].fillna(s3[fea].median(), inplace=True)
        for fea in s3.columns:
                s3[fea].fillna(0, inplace=True)
        s4=s4.drop(['price_doc','sub_area','product_type'],axis=1)
        s4=s4.iloc[:,2:20]
        for fea in s4.columns:
                s4[fea].fillna(s4[fea].median(), inplace=True)
        for fea in s4.columns:
                s4[fea].fillna(0, inplace=True)
        
        sa=sa.drop(['price_doc','sub_area','product_type'],axis=1)
        sa=sa.iloc[:,2:20]
        for fea in sa.columns:
                sa[fea].fillna(sa[fea].median(), inplace=True)
        for fea in sa.columns:
                sa[fea].fillna(0, inplace=True)
                
                
        if self.FLAG1==1:
           X23 = pd.concat([s2,s3]) 
           y23 = np.concatenate([t2,t3],axis=0)
           random_estimator23=RandomForestRegressor(random_state=0, n_estimators=500)
           random_estimator23.fit(X23,y23)
           self.random_estimator2=random_estimator23
           self.random_estimator3=random_estimator23
           
           X23r=X23.full_sq
           X23r=X23r.reshape(len(X23r),1)
           y23r=y23.reshape(len(y23), 1)
           Linear_estimator23=LinearRegression()
           Linear_estimator23.fit(X23r, y23r)
           
           self.Linear_estimator2=Linear_estimator23
           self.Linear_estimator3=Linear_estimator23
           
        else:
            
            random_estimator2 = RandomForestRegressor(random_state=0, n_estimators=500)
            random_estimator2.fit(s2,t2)
            self.random_estimator2=random_estimator2
            
            Linear_estimator2 = LinearRegression()
            
            s2r=s2.full_sq
            s2r=s2r.reshape(len(s2r),1)
            t2r=t2
            t2r=t2r.reshape(len(t2r),1)
            Linear_estimator2.fit(s2r, t2r)
            self.Linear_estimator2=Linear_estimator2
            
            
            
            random_estimator3 = RandomForestRegressor(random_state=0, n_estimators=500)
            random_estimator3.fit(s3,t3)
            self.random_estimator3=random_estimator3
            
            Linear_estimator3 = LinearRegression()
            
            s3r=s3.full_sq
            s3r=s3r.reshape(len(s3r),1)
            t3r=t3
            t3r=t3r.reshape(len(t3r),1)
            Linear_estimator3.fit(s3r, t3r)
            self.Linear_estimator3=Linear_estimator3
            
            
            
           
        if self.FLAG2==1:
               self.TwoClassModel=twoclass
        else :
               
                    
                d3=np.ones(len(t3))
                d4=np.ones(len(t4))*-1
                          
                X = pd.concat([s3,s4])
                y = np.concatenate([d3,d4],axis=0)
                self.y=y
                self.X=X
                #print (X)
                #self.decision_clf = svm.SVC()
                TwoClassModel = RandomForestClassifier()
                TwoClassModel.fit(X, y)
                self.TwoClassModel=TwoClassModel
                
        
        random_estimator4 = RandomForestRegressor(random_state=0, n_estimators=500)
        random_estimator4.fit(s4,t4)
        self.random_estimator4=random_estimator4
        
        Linear_estimator4 = LinearRegression()        
        s4r=s4.full_sq
        s4r=s4r.reshape(len(s4r),1)
        t4r=t4
        t4r=t4r.reshape(len(t4r),1)
        Linear_estimator4.fit(s4r, t4r)
        self.Linear_estimator4=Linear_estimator4             

        
    def predict_price(self,test_row):
        testrow1=test_row
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
        if dum > self.highcutoff1:
            #predicted_price=5
            predicted_price1= self.random_estimator3.predict(test_row)[0] 
            
            x1 = test_row.full_sq
            x1 = x1.reshape(len(x1),1)
            
            predicted_price2= self.Linear_estimator3.predict(x1)[0]
            
        
            predicted_price=0.2*predicted_price1+ 0.8*predicted_price2
            
            
        else:
            if flag==1:
                
                predicted_price1= self.random_estimator2.predict(test_row)[0] 
                x1 = test_row.full_sq
                x1 = x1.reshape(len(x1),1)
                predicted_price2= self.Linear_estimator2.predict(x1)[0]
                predicted_price=0.2*predicted_price1+ 0.8*predicted_price2
            
            else:
                if self.FLAG2==1:
                    testrow1=testrow1.drop(['sub_area', 'full_sq' ,'product_type'],axis=1)
                    testrow1=testrow1.iloc[:,2:20]
                    for fea in testrow1.columns:
                              testrow1[fea].fillna(testrow1[fea].median(),inplace=True)
                    for fea in testrow1.columns:
                              testrow1[fea].fillna(0,inplace=True)
                              
                    pclass=self.TwoClassModel.predict(testrow1)[0]
                
                else:
                       pclass=self.TwoClassModel.predict(test_row)[0]
                       
                if pclass==-1:
                       #predicted_price= 1.5
                       predicted_price1= self.random_estimator4.predict(test_row)[0]
                       x1 = test_row.full_sq
                       x1 = x1.reshape(len(x1),1)
                       predicted_price2= self.Linear_estimator4.predict(x1)[0]
                       predicted_price=0.9*predicted_price1+ 0.1*predicted_price2
                else:                        
                       predicted_price1= self.random_estimator3.predict(test_row)[0] 
                       x1 = test_row.full_sq
                       x1 = x1.reshape(len(x1),1)
                       predicted_price2= self.Linear_estimator3.predict(x1)[0]
                       predicted_price=0.2*predicted_price1+ 0.8*predicted_price2
                       
        if self.specialFlag==1:
            if dum < 65 :
                 predicted_price=predicted_price-2
      
        return predicted_price
#################################################################

model_146=small_models   ## 'Poselenie Mihajlovo-Jarcevskoe'
model_145=small_models   ##  'Molzhaninovskoe'
model_144=small_models   ## 'Poselenie Kievskij'
model_143=small_models   ##  'Poselenie Klenovskoe'
model_142=small_models   ## 'Poselenie Voronovskoe'
model_141=small_models   ## 'Vostochnoe'

model_140=Poselenie_Kokoshkino(train_df[train_df.sub_area=='Poselenie Kokoshkino'], TwoClassModel)
model_139=model_all
model_138=small_models   ## 'Poselenie Shhapovskoe'
model_137=small_models   ## 'Poselenie Marushkinskoe'
model_136=model_all

model_135=Poselenie_Pervomajskoe(train_df[train_df.sub_area=='Poselenie Pervomajskoe'], TwoClassModel)
model_134=Metrogorodok(train_df[train_df.sub_area=='Metrogorodok'], TwoClassModel)
model_133=model_all
model_132=LinearModel(train_df[train_df.sub_area=='Sokol\'niki'], TwoClassModel)
model_131=LinearModel(train_df[train_df.sub_area=='Savelki'], TwoClassModel)
model_130=LinearModel(train_df[train_df.sub_area=='Nagatinskij Zaton'], TwoClassModel)


model_129=LinearModel(train_df[train_df.sub_area=='Prospekt Vernadskogo'], TwoClassModel)
model_128=LinearModel(train_df[train_df.sub_area=='Levoberezhnoe'], TwoClassModel)
model_127=LinearModel(train_df[train_df.sub_area=='Begovoe'], TwoClassModel)
model_126=LinearModel(train_df[train_df.sub_area=='Vnukovo'], TwoClassModel)
model_125=model_all # 'Marfino'
model_124=LinearModel(train_df[train_df.sub_area=='Cheremushki'], TwoClassModel)
model_123=LinearModel(train_df[train_df.sub_area=='Dorogomilovo'], TwoClassModel)
model_122=LinearModel(train_df[train_df.sub_area=='Troparevo-Nikulino'], TwoClassModel)
model_121=LinearModel(train_df[train_df.sub_area=='Babushkinskoe'], TwoClassModel)
model_120=LinearModel(train_df[train_df.sub_area=='Gagarinskoe'], TwoClassModel)

model_119=LinearModel(train_df[train_df.sub_area=='Brateevo'], TwoClassModel)
model_118=LinearModel(train_df[train_df.sub_area=='Koptevo'], TwoClassModel)
model_117=LinearModel(train_df[train_df.sub_area=='Strogino'], TwoClassModel)
model_116=LinearModel(train_df[train_df.sub_area=='Ljublino'], TwoClassModel)
model_115=LinearModel(train_df[train_df.sub_area=='Lomonosovskoe'], TwoClassModel)
model_114=LinearModel(train_df[train_df.sub_area=='Nagornoe'], TwoClassModel)
model_113=LinearModel(train_df[train_df.sub_area=='Teplyj Stan'], TwoClassModel)
model_112=LinearModel(train_df[train_df.sub_area=='Moskvorech\'e-Saburovo'], TwoClassModel)
model_111=LinearModel(train_df[train_df.sub_area=='Juzhnoportovoe'], TwoClassModel)
model_110=LinearModel(train_df[train_df.sub_area=='Fili Davydkovo'], TwoClassModel)
model_109=LinearModel(train_df[train_df.sub_area=='Hamovniki'], TwoClassModel)
model_108=LinearModel(train_df[train_df.sub_area=='Juzhnoe Tushino'], TwoClassModel)
model_107=LinearModel(train_df[train_df.sub_area=='Preobrazhenskoe'], TwoClassModel)
model_106=LinearModel(train_df[train_df.sub_area=='Izmajlovo'], TwoClassModel)
model_105=LinearModel(train_df[train_df.sub_area=='Orehovo-Borisovo Severnoe'], TwoClassModel)
model_104=LinearModel(train_df[train_df.sub_area=='Vostochnoe Izmajlovo'], TwoClassModel)
model_103=model_all # 'Mar\'ina Roshha'
model_102=LinearModel(train_df[train_df.sub_area=='Losinoostrovskoe'], TwoClassModel)
model_101=LinearModel(train_df[train_df.sub_area=='Chertanovo Central\'noe'], TwoClassModel)

model_18=small_models # 'Poselenie Krasnopahorskoe'
model_15=small_models # 'Poselenie Rogovskoe'


###################################################################
#####################################################################

################################################################################################################


model_1=Chertanovo_Severnoe(train_df[train_df.sub_area=='Chertanovo Severnoe'],TwoClassModel) #done
model_2=Krjukovo(train_df[train_df.sub_area=='Krjukovo'],TwoClassModel)                       # done
model_3=Rostokino(train_df[train_df.sub_area=='Rostokino'],TwoClassModel) #done
model_4=Basmannoe(train_df[train_df.sub_area=='Basmannoe'],TwoClassModel) #done
model_5=Poselenie_Vnukovskoe(train_df[train_df.sub_area=='Poselenie Vnukovskoe'],TwoClassModel) #done
model_6=Poselenie_Desjonovskoe(train_df[train_df.sub_area=='Poselenie Desjonovskoe'],TwoClassModel) #done check heigher values in train
model_7=Poselenie_Sosenskoe(train_df[train_df.sub_area=='Poselenie Sosenskoe'],TwoClassModel) # done good model fractional model (check high values)
model_8=Poselenie_Filimonkovskoe(train_df[train_df.sub_area=='Poselenie Filimonkovskoe'],TwoClassModel)
model_9=Poselenie_Voskresenskoe(train_df[train_df.sub_area=='Poselenie Voskresenskoe'],TwoClassModel)
model_10=Novo_Peredelkino(train_df[train_df.sub_area=='Novo-Peredelkino'],TwoClassModel)
model_11=Poselenie_Moskovskij(train_df[train_df.sub_area=='Poselenie Moskovskij'],TwoClassModel)
model_12=Poselenie_Shherbinka(train_df[train_df.sub_area=='Poselenie Shherbinka'],TwoClassModel)
model_13=Nekrasovka(train_df[train_df.sub_area=='Nekrasovka'],TwoClassModel)
model_14=Akademicheskoe(train_df[train_df.sub_area=='Akademicheskoe'],TwoClassModel)
#model_=(train_df[train_df.sub_area==''],TwoClassModel)
model_16=Sokol(train_df[train_df.sub_area=='Sokol'],TwoClassModel)
model_17=Kuzminki(train_df[train_df.sub_area=="Kuz'minki"],TwoClassModel)
#model_=(train_df[train_df.sub_area==''],TwoClassModel)
model_19=Zjuzino(train_df[train_df.sub_area=='Zjuzino'],TwoClassModel)
model_20=Poselenie_Novofedorovskoe(train_df[train_df.sub_area=='Poselenie_Novofedorovskoe'],TwoClassModel)
model_21=Juzhnoe_Butovo(train_df[train_df.sub_area=='Juzhnoe Butovo'],TwoClassModel)
model_22=Matushkino(train_df[train_df.sub_area=='Matushkino'],TwoClassModel)
model_23=Otradnoe(train_df[train_df.sub_area=='Otradnoe'],TwoClassModel)
model_24=Mitino(train_df[train_df.sub_area=='Mitino'],TwoClassModel)
model_25=Beskudnikovskoe(train_df[train_df.sub_area=='Beskudnikovskoe'],TwoClassModel)
model_26=Vojkovskoe(train_df[train_df.sub_area=='Vojkovskoe'],TwoClassModel)
model_27=Troickij_okrug(train_df[train_df.sub_area=='Troickij okrug'],TwoClassModel)
model_28=Rjazanskij(train_df[train_df.sub_area=='Rjazanskij'],TwoClassModel)
model_29=Severnoe_Butovo(train_df[train_df.sub_area=='Severnoe Butovo'],TwoClassModel)
model_30=Kuncevo(train_df[train_df.sub_area=='Kuncevo'],TwoClassModel)
model_31=Severnoe(train_df[train_df.sub_area=='Severnoe'],TwoClassModel)
model_32=Staroe_Krjukovo(train_df[train_df.sub_area=='Staroe Krjukovo'],TwoClassModel)
model_33=Golovinskoe(train_df[train_df.sub_area=='Golovinskoe'],TwoClassModel)
model_34=Konkovo(train_df[train_df.sub_area=="Kon'kovo"],TwoClassModel)
model_35=Kosino_Uhtomskoe(train_df[train_df.sub_area=='Kosino-Uhtomskoe'],TwoClassModel)
model_36=Veshnjaki(train_df[train_df.sub_area=='Veshnjaki'],TwoClassModel)
model_37=Horoshevo_Mnevniki(train_df[train_df.sub_area=='Horoshevo-Mnevniki'],TwoClassModel)
model_38=Marino(train_df[train_df.sub_area=="Mar'ino"],TwoClassModel)
model_39=Pechatniki(train_df[train_df.sub_area=="Pechatniki"],TwoClassModel)
model_40=Tekstilshhiki(train_df[train_df.sub_area=="Tekstil'shhiki"],TwoClassModel)
model_41=Solncevo(train_df[train_df.sub_area=="Solncevo"],TwoClassModel)
model_42=Sviblovo(train_df[train_df.sub_area=="Sviblovo"],TwoClassModel)
model_43=Silino(train_df[train_df.sub_area=="Silino"],TwoClassModel)
model_44=Obruchevskoe(train_df[train_df.sub_area=="Obruchevskoe"],TwoClassModel)
model_45=Butyrskoe(train_df[train_df.sub_area=="Butyrskoe"],TwoClassModel)
model_46=Birjulevo_Vostochnoe(train_df[train_df.sub_area=="Birjulevo Vostochnoe"],TwoClassModel)
model_47=Caricyno(train_df[train_df.sub_area=="Caricyno"],TwoClassModel)
model_48=Krylatskoe(train_df[train_df.sub_area=="Krylatskoe"],TwoClassModel)
model_49=Shhukino(train_df[train_df.sub_area=="Shhukino"],TwoClassModel)
model_50=Tverskoe(train_df[train_df.sub_area=="Tverskoe"],TwoClassModel)
model_51=Taganskoe(train_df[train_df.sub_area=="Taganskoe"],TwoClassModel)
model_52=Kapotnja(train_df[train_df.sub_area=="Kapotnja"],TwoClassModel)
model_53=Horoshevskoe(train_df[train_df.sub_area=="Horoshevskoe"],TwoClassModel)
model_54=Orehovo_Borisovo_Juzhnoe(train_df[train_df.sub_area=="Orehovo-Borisovo Juzhnoe"],TwoClassModel)
model_55=Krasnoselskoe(train_df[train_df.sub_area=="Krasnosel'skoe"],TwoClassModel)
model_56=Zamoskvoreche(train_df[train_df.sub_area=="Zamoskvorech'e"],TwoClassModel)
model_57=Dmitrovskoe(train_df[train_df.sub_area=="Dmitrovskoe"],TwoClassModel)
model_58=Mozhajskoe(train_df[train_df.sub_area=="Mozhajskoe"],TwoClassModel)
model_59=Juzhnoe_Medvedkovo(train_df[train_df.sub_area=="Juzhnoe Medvedkovo"],TwoClassModel)
model_60=Sokolinaja_Gora(train_df[train_df.sub_area=="Sokolinaja Gora"],TwoClassModel)
model_61=Lianozovo(train_df[train_df.sub_area=="Lianozovo"],TwoClassModel)
model_62=Pokrovskoe_Streshnevo(train_df[train_df.sub_area=="Pokrovskoe Streshnevo"],TwoClassModel)
model_62=Zapadnoe_Degunino(train_df[train_df.sub_area=="Zapadnoe Degunino"],TwoClassModel)
model_63=Ajeroport(train_df[train_df.sub_area=="Ajeroport"],TwoClassModel)
model_64=Novogireevo(train_df[train_df.sub_area=="Novogireevo"],TwoClassModel)
model_65=Goljanovo(train_df[train_df.sub_area=="Gol'janovo"],TwoClassModel)
model_66=Ostankinskoe(train_df[train_df.sub_area=="Ostankinskoe"],TwoClassModel)
model_67=Bogorodskoe(train_df[train_df.sub_area=="Bogorodskoe"],TwoClassModel)
model_68=Savelovskoe(train_df[train_df.sub_area=="Savelovskoe"],TwoClassModel)
model_69=Presnenskoe(train_df[train_df.sub_area=="Presnenskoe"],TwoClassModel)
model_70=Presnenskoe(train_df[train_df.sub_area=="Presnenskoe"],TwoClassModel)
model_71=Alekseevskoe(train_df[train_df.sub_area=="Alekseevskoe"],TwoClassModel)
model_72=Timirjazevskoe(train_df[train_df.sub_area=="Timirjazevskoe"],TwoClassModel)
model_73=Jasenevo(train_df[train_df.sub_area=="Jasenevo"],TwoClassModel)          #check 
model_74=Altufevskoe(train_df[train_df.sub_area=="Altuf'evskoe"],TwoClassModel)
model_75=Severnoe_Medvedkovo(train_df[train_df.sub_area=="Severnoe Medvedkovo"],TwoClassModel)
model_76=Kurkino(train_df[train_df.sub_area=="Kurkino"],TwoClassModel)
model_77=Vyhino_Zhulebino(train_df[train_df.sub_area=="Vyhino-Zhulebino"],TwoClassModel)
model_78=Filevskij_Park(train_df[train_df.sub_area=="Filevskij Park"],TwoClassModel)
model_79=Kotlovka(train_df[train_df.sub_area=="Kotlovka"],TwoClassModel)
model_80=Lefortovo(train_df[train_df.sub_area=="Lefortovo"],TwoClassModel)
model_81=Severnoe_Izmajlovo(train_df[train_df.sub_area=="Severnoe Izmajlovo"],TwoClassModel)
model_82=Ochakovo_Matveevskoe(train_df[train_df.sub_area=="Ochakovo-Matveevskoe"],TwoClassModel)
model_83=Ochakovo_Matveevskoe(train_df[train_df.sub_area=="Ochakovo-Matveevskoe"],TwoClassModel)
model_84=Perovo(train_df[train_df.sub_area=="Perovo"],TwoClassModel)
model_85=Nizhegorodskoe(train_df[train_df.sub_area=="Nizhegorodskoe"],TwoClassModel)
model_86=Jakimanka(train_df[train_df.sub_area=="Jakimanka"],TwoClassModel)
model_87=Ivanovskoe(train_df[train_df.sub_area=="Ivanovskoe"],TwoClassModel)
model_88=Severnoe_Tushino(train_df[train_df.sub_area=="Severnoe Tushino"],TwoClassModel)
model_89=Nagatino_Sadovniki(train_df[train_df.sub_area=="Nagatino-Sadovniki"],TwoClassModel)
model_90=Ramenki(train_df[train_df.sub_area=="Ramenki"],TwoClassModel)
model_91=Bibirevo(train_df[train_df.sub_area=="Bibirevo"],TwoClassModel)
model_92=Zjablikovo(train_df[train_df.sub_area=="Zjablikovo"],TwoClassModel)
model_93=Meshhanskoe(train_df[train_df.sub_area=="Meshhanskoe"],TwoClassModel)
model_94=Chertanovo_Juzhnoe(train_df[train_df.sub_area=="Chertanovo Juzhnoe"],TwoClassModel)
model_95=Danilovskoe(train_df[train_df.sub_area=="Danilovskoe"],TwoClassModel)
model_96=Hovrino(train_df[train_df.sub_area=="Hovrino"],TwoClassModel)
model_98=Novokosino(train_df[train_df.sub_area=="Novokosino"],TwoClassModel)
model_97=Vostochnoe_Degunino(train_df[train_df.sub_area=="Vostochnoe Degunino"],TwoClassModel)
model_99=Birjulevo_Zapadnoe(train_df[train_df.sub_area=="Birjulevo Zapadnoe"],TwoClassModel)
model_100=Donskoe(train_df[train_df.sub_area=="Donskoe"],TwoClassModel)

#model_=(train_df[train_df.sub_area==""],TwoClassModel)
#model_=(train_df[train_df.sub_area==""],TwoClassModel)
#model_=(train_df[train_df.sub_area==""],TwoClassModel)
#model_=(train_df[train_df.sub_area==""],TwoClassModel)
#model_=(train_df[train_df.sub_area==""],TwoClassModel)
#model_=(train_df[train_df.sub_area==""],TwoClassModel)
#model_=(train_df[train_df.sub_area==""],TwoClassModel)
#model_=(train_df[train_df.sub_area==""],TwoClassModel)
#model_=(train_df[train_df.sub_area==""],TwoClassModel)
#model_=(train_df[train_df.sub_area==""],TwoClassModel)
#model_=(train_df[train_df.sub_area==""],TwoClassModel)

#model_1=Chertanovo_Severnoe(train_df[train_df.sub_area=='Chertanovo Severnoe'])
i=0
predicted=[]



for pos, row in test_df.iterrows():
    flag1=0
    y=0
    if row.sub_area=="Novo-Peredelkino": 
        y=model_10.predict_price(pd.DataFrame(row).T)
 
        flag1 += 1 
           
    
    if row.sub_area=="Donskoe": 
        y=model_100.predict_price(pd.DataFrame(row).T)
 
        flag1 += 1 
   
    
    if row.sub_area=="Poselenie Moskovskij": 
        y=model_11.predict_price(pd.DataFrame(row).T)
 
        flag1 += 1 
   
    
    if row.sub_area=="Poselenie Shherbinka": 
        y=model_12.predict_price(pd.DataFrame(row).T)
 
        flag1 += 1 
   
    
    if row.sub_area=="Nekrasovka": 
        y=model_13.predict_price(pd.DataFrame(row).T)
 
        flag1 += 1 
   
    
    if row.sub_area=="Akademicheskoe": 
        y=model_14.predict_price(pd.DataFrame(row).T)
 
        flag1 += 1 
   
    
    if row.sub_area=="Sokol": 
        y=model_16.predict_price(pd.DataFrame(row).T)
 
        flag1 += 1 
   
    
    if row.sub_area=="Kuz'minki": 
        y=model_17.predict_price(pd.DataFrame(row).T)
 
        flag1 += 1 
   
    
    if row.sub_area=="Zjuzino": 
        y=model_19.predict_price(pd.DataFrame(row).T)
 
        flag1 += 1 
   
    
    if row.sub_area=="Krjukovo": 
        y=model_2.predict_price(pd.DataFrame(row).T)
 
        flag1 += 1 
   
    
    if row.sub_area=="Poselenie_Novofedorovskoe": 
        y=model_20.predict_price(pd.DataFrame(row).T)
 
        flag1 += 1 
   
    
    if row.sub_area=="Juzhnoe Butovo": 
        y=model_21.predict_price(pd.DataFrame(row).T)
 
        flag1 += 1 
   
    
    if row.sub_area=="Matushkino": 
        y=model_22.predict_price(pd.DataFrame(row).T)
 
        flag1 += 1 
   
    
    if row.sub_area=="Otradnoe": 
        y=model_23.predict_price(pd.DataFrame(row).T)
 
        flag1 += 1 
   
    
    if row.sub_area=="Mitino": 
        y=model_24.predict_price(pd.DataFrame(row).T)
 
        flag1 += 1 
   
    
    if row.sub_area=="Beskudnikovskoe": 
        y=model_25.predict_price(pd.DataFrame(row).T)
 
        flag1 += 1 
   
    
    if row.sub_area=="Vojkovskoe": 
        y=model_26.predict_price(pd.DataFrame(row).T)
 
        flag1 += 1 
   
    
    if row.sub_area=="Troickij okrug": 
        y=model_27.predict_price(pd.DataFrame(row).T)
 
        flag1 += 1 
   
    
    if row.sub_area=="Rjazanskij": 
        y=model_28.predict_price(pd.DataFrame(row).T)
 
        flag1 += 1 
   
    
    if row.sub_area=="Severnoe Butovo": 
        y=model_29.predict_price(pd.DataFrame(row).T)
 
        flag1 += 1 
   
    
    if row.sub_area=="Rostokino": 
        y=model_3.predict_price(pd.DataFrame(row).T)
 
        flag1 += 1 
   
    
    if row.sub_area=="Kuncevo": 
        y=model_30.predict_price(pd.DataFrame(row).T)
 
        flag1 += 1 
   
    
    if row.sub_area=="Severnoe": 
        y=model_31.predict_price(pd.DataFrame(row).T)
 
        flag1 += 1 
   
    
    if row.sub_area=="Staroe Krjukovo": 
        y=model_32.predict_price(pd.DataFrame(row).T)
 
        flag1 += 1 
   
    
    if row.sub_area=="Golovinskoe": 
        y=model_33.predict_price(pd.DataFrame(row).T)
 
        flag1 += 1 
   
    
    if row.sub_area=="Kon'kovo": 
        y=model_34.predict_price(pd.DataFrame(row).T)
 
        flag1 += 1 
   
    
    if row.sub_area=="Kosino-Uhtomskoe": 
        y=model_35.predict_price(pd.DataFrame(row).T)
 
        flag1 += 1 
   
    
    if row.sub_area=="Veshnjaki": 
        y=model_36.predict_price(pd.DataFrame(row).T)
 
        flag1 += 1 
   
    
    if row.sub_area=="Horoshevo-Mnevniki": 
        y=model_37.predict_price(pd.DataFrame(row).T)
 
        flag1 += 1 
   
    
    if row.sub_area=="Mar'ino": 
        y=model_38.predict_price(pd.DataFrame(row).T)
 
        flag1 += 1 
   
    
    if row.sub_area=="Pechatniki": 
        y=model_39.predict_price(pd.DataFrame(row).T)
 
        flag1 += 1 
   
    
    if row.sub_area=="Basmannoe": 
        y=model_4.predict_price(pd.DataFrame(row).T)
 
        flag1 += 1 
   
    
    if row.sub_area=="Tekstil'shhiki": 
        y=model_40.predict_price(pd.DataFrame(row).T)
 
        flag1 += 1 
   
    
    if row.sub_area=="Solncevo": 
        y=model_41.predict_price(pd.DataFrame(row).T)
 
        flag1 += 1 
   
    
    if row.sub_area=="Sviblovo": 
        y=model_42.predict_price(pd.DataFrame(row).T)
 
        flag1 += 1 
   
    
    if row.sub_area=="Silino": 
        y=model_43.predict_price(pd.DataFrame(row).T)
 
        flag1 += 1 
   
    
    if row.sub_area=="Obruchevskoe": 
        y=model_44.predict_price(pd.DataFrame(row).T)
 
        flag1 += 1 
   
    
    if row.sub_area=="Butyrskoe": 
        y=model_45.predict_price(pd.DataFrame(row).T)
 
        flag1 += 1 
   
    
    if row.sub_area=="Birjulevo Vostochnoe": 
        y=model_46.predict_price(pd.DataFrame(row).T)
 
        flag1 += 1 
   
    
    if row.sub_area=="Caricyno": 
        y=model_47.predict_price(pd.DataFrame(row).T)
 
        flag1 += 1 
   
    
    if row.sub_area=="Krylatskoe": 
        y=model_48.predict_price(pd.DataFrame(row).T)
 
        flag1 += 1 
   
    
    if row.sub_area=="Shhukino": 
        y=model_49.predict_price(pd.DataFrame(row).T)
 
        flag1 += 1 
   
    
    if row.sub_area=="Poselenie Vnukovskoe": 
        y=model_5.predict_price(pd.DataFrame(row).T)
 
        flag1 += 1 
   
    
    if row.sub_area=="Tverskoe": 
        y=model_50.predict_price(pd.DataFrame(row).T)
 
        flag1 += 1 
   
    
    if row.sub_area=="Taganskoe": 
        y=model_51.predict_price(pd.DataFrame(row).T)
 
        flag1 += 1 
   
    
    if row.sub_area=="Kapotnja": 
        y=model_52.predict_price(pd.DataFrame(row).T)
 
        flag1 += 1 
   
    
    if row.sub_area=="Horoshevskoe": 
        y=model_53.predict_price(pd.DataFrame(row).T)
 
        flag1 += 1 
   
    
    if row.sub_area=="Orehovo-Borisovo Juzhnoe": 
        y=model_54.predict_price(pd.DataFrame(row).T)
 
        flag1 += 1 
   
    
    if row.sub_area=="Krasnosel'skoe": 
        y=model_55.predict_price(pd.DataFrame(row).T)
 
        flag1 += 1 
   
    
    if row.sub_area=="Zamoskvorech'e": 
        y=model_56.predict_price(pd.DataFrame(row).T)
 
        flag1 += 1 
   
    
    if row.sub_area=="Dmitrovskoe": 
        y=model_57.predict_price(pd.DataFrame(row).T)
 
        flag1 += 1 
   
    
    if row.sub_area=="Mozhajskoe": 
        y=model_58.predict_price(pd.DataFrame(row).T)
 
        flag1 += 1 
   
    
    if row.sub_area=="Juzhnoe Medvedkovo": 
        y=model_59.predict_price(pd.DataFrame(row).T)
 
        flag1 += 1 
   
    
    if row.sub_area=="Poselenie Desjonovskoe": 
        y=model_6.predict_price(pd.DataFrame(row).T)
 
        flag1 += 1 
   
    
    if row.sub_area=="Sokolinaja Gora": 
        y=model_60.predict_price(pd.DataFrame(row).T)
 
        flag1 += 1 
   
    
    if row.sub_area=="Lianozovo": 
        y=model_61.predict_price(pd.DataFrame(row).T)
 
        flag1 += 1 
   
    
    if row.sub_area=="Pokrovskoe Streshnevo": 
        y=model_62.predict_price(pd.DataFrame(row).T)
 
        flag1 += 1 
   
    
    if row.sub_area=="Ajeroport": 
        y=model_63.predict_price(pd.DataFrame(row).T)
 
        flag1 += 1 
   
    
    if row.sub_area=="Novogireevo": 
        y=model_64.predict_price(pd.DataFrame(row).T)
 
        flag1 += 1 
   
    
    if row.sub_area=="Gol'janovo": 
        y=model_65.predict_price(pd.DataFrame(row).T)
 
        flag1 += 1 
   
    
    if row.sub_area=="Ostankinskoe": 
        y=model_66.predict_price(pd.DataFrame(row).T)
 
        flag1 += 1 
   
    
    if row.sub_area=="Bogorodskoe": 
        y=model_67.predict_price(pd.DataFrame(row).T)
 
        flag1 += 1 
   
    
    if row.sub_area=="Savelovskoe": 
        y=model_68.predict_price(pd.DataFrame(row).T)
 
        flag1 += 1 
   
    
    if row.sub_area=="Presnenskoe": 
        y=model_69.predict_price(pd.DataFrame(row).T)
 
        flag1 += 1 
   
    
    if row.sub_area=="Poselenie Sosenskoe": 
        y=model_7.predict_price(pd.DataFrame(row).T)
 
        flag1 += 1 
   
    
    if row.sub_area=="Presnenskoe": 
        y=model_70.predict_price(pd.DataFrame(row).T)
 
        flag1 += 1 
   
    
    if row.sub_area=="Alekseevskoe": 
        y=model_71.predict_price(pd.DataFrame(row).T)
 
        flag1 += 1 
   
    
    if row.sub_area=="Timirjazevskoe": 
        y=model_72.predict_price(pd.DataFrame(row).T)
 
        flag1 += 1 
   
    
    if row.sub_area=="Jasenevo": 
        y=model_73.predict_price(pd.DataFrame(row).T)
 
        flag1 += 1 
   
    
    if row.sub_area=="Altuf'evskoe": 
        y=model_74.predict_price(pd.DataFrame(row).T)
 
        flag1 += 1 
   
    
    if row.sub_area=="Severnoe Medvedkovo": 
        y=model_75.predict_price(pd.DataFrame(row).T)
 
        flag1 += 1 
   
    
    if row.sub_area=="Kurkino": 
        y=model_76.predict_price(pd.DataFrame(row).T)
 
        flag1 += 1 
   
    
    if row.sub_area=="Vyhino-Zhulebino": 
        y=model_77.predict_price(pd.DataFrame(row).T)
 
        flag1 += 1 
   
    
    if row.sub_area=="Filevskij Park": 
        y=model_78.predict_price(pd.DataFrame(row).T)
 
        flag1 += 1 
   
    
    if row.sub_area=="Kotlovka": 
        y=model_79.predict_price(pd.DataFrame(row).T)
 
        flag1 += 1 
   
    
    if row.sub_area=="Poselenie Filimonkovskoe": 
        y=model_8.predict_price(pd.DataFrame(row).T)
 
        flag1 += 1 
   
    
    if row.sub_area=="Lefortovo": 
        y=model_80.predict_price(pd.DataFrame(row).T)
 
        flag1 += 1 
   
    
    if row.sub_area=="Severnoe Izmajlovo": 
        y=model_81.predict_price(pd.DataFrame(row).T)
 
        flag1 += 1 
   
    
    if row.sub_area=="Ochakovo-Matveevskoe": 
        y=model_82.predict_price(pd.DataFrame(row).T)
 
        flag1 += 1 
   
    
    if row.sub_area=="Ochakovo-Matveevskoe": 
        y=model_83.predict_price(pd.DataFrame(row).T)
 
        flag1 += 1 
   
    
    if row.sub_area=="Perovo": 
        y=model_84.predict_price(pd.DataFrame(row).T)
 
        flag1 += 1 
   
    
    if row.sub_area=="Nizhegorodskoe": 
        y=model_85.predict_price(pd.DataFrame(row).T)
 
        flag1 += 1 
   
    
    if row.sub_area=="Jakimanka": 
        y=model_86.predict_price(pd.DataFrame(row).T)
 
        flag1 += 1 
   
    
    if row.sub_area=="Ivanovskoe": 
        y=model_87.predict_price(pd.DataFrame(row).T)
 
        flag1 += 1 
   
    
    if row.sub_area=="Severnoe Tushino": 
        y=model_88.predict_price(pd.DataFrame(row).T)
 
        flag1 += 1 
   
    
    if row.sub_area=="Nagatino-Sadovniki": 
        y=model_89.predict_price(pd.DataFrame(row).T)
 
        flag1 += 1 
   
    
    if row.sub_area=="Poselenie Voskresenskoe": 
        y=model_9.predict_price(pd.DataFrame(row).T)
 
        flag1 += 1 
   
    
    if row.sub_area=="Ramenki": 
        y=model_90.predict_price(pd.DataFrame(row).T)
 
        flag1 += 1 
   
    
    if row.sub_area=="Bibirevo": 
        y=model_91.predict_price(pd.DataFrame(row).T)
 
        flag1 += 1 
   
    
    if row.sub_area=="Zjablikovo": 
        y=model_92.predict_price(pd.DataFrame(row).T)
 
        flag1 += 1 
   
    
    if row.sub_area=="Meshhanskoe": 
        y=model_93.predict_price(pd.DataFrame(row).T)
 
        flag1 += 1 
   
    
    if row.sub_area=="Chertanovo Juzhnoe": 
        y=model_94.predict_price(pd.DataFrame(row).T)
 
        flag1 += 1 
   
    
    if row.sub_area=="Danilovskoe": 
        y=model_95.predict_price(pd.DataFrame(row).T)
 
        flag1 += 1 
   
    
    if row.sub_area=="Hovrino": 
        y=model_96.predict_price(pd.DataFrame(row).T)
 
        flag1 += 1 
   
    
    if row.sub_area=="Vostochnoe Degunino": 
        y=model_97.predict_price(pd.DataFrame(row).T)
 
        flag1 += 1 
   
    
    if row.sub_area=="Novokosino": 
        y=model_98.predict_price(pd.DataFrame(row).T)
 
        flag1 += 1 
   
    
    if row.sub_area=="Birjulevo Zapadnoe": 
        y=model_99.predict_price(pd.DataFrame(row).T)
 
        flag1 += 1 
   
    
    if row.sub_area=="Chertanovo Central'noe": 
        y=model_101.predict_price(pd.DataFrame(row).T)
 
        flag1 += 1 
   
    
    if row.sub_area=="Losinoostrovskoe": 
        y=model_102.predict_price(pd.DataFrame(row).T)
 
        flag1 += 1 
   
    
    if row.sub_area=="Mar'ina Roshha": 
        y=model_103.predict_price(pd.DataFrame(row).T)
 
        flag1 += 1 
   
    
    if row.sub_area=="Vostochnoe Izmajlovo": 
        y=model_104.predict_price(pd.DataFrame(row).T)
 
        flag1 += 1 
   
    
    if row.sub_area=="Orehovo-Borisovo Severnoe": 
        y=model_105.predict_price(pd.DataFrame(row).T)
 
        flag1 += 1 
   
    
    if row.sub_area=="Izmajlovo": 
        y=model_106.predict_price(pd.DataFrame(row).T)
 
        flag1 += 1 
   
    
    if row.sub_area=="Preobrazhenskoe": 
        y=model_107.predict_price(pd.DataFrame(row).T)
 
        flag1 += 1 
   
    
    if row.sub_area=="Juzhnoe Tushino": 
        y=model_108.predict_price(pd.DataFrame(row).T)
 
        flag1 += 1 
   
    
    if row.sub_area=="Hamovniki": 
        y=model_109.predict_price(pd.DataFrame(row).T)
 
        flag1 += 1 
   
    
    if row.sub_area=="Fili Davydkovo": 
        y=model_110.predict_price(pd.DataFrame(row).T)
 
        flag1 += 1 
   
    
    if row.sub_area=="Juzhnoportovoe": 
        y=model_111.predict_price(pd.DataFrame(row).T)
 
        flag1 += 1 
   
    
    if row.sub_area=="Moskvorech'e-Saburovo": 
        y=model_112.predict_price(pd.DataFrame(row).T)
 
        flag1 += 1 
   
    
    if row.sub_area=="Teplyj Stan": 
        y=model_113.predict_price(pd.DataFrame(row).T)
 
        flag1 += 1 
   
    
    if row.sub_area=="Nagornoe": 
        y=model_114.predict_price(pd.DataFrame(row).T)
 
        flag1 += 1 
   
    
    if row.sub_area=="Lomonosovskoe": 
        y=model_115.predict_price(pd.DataFrame(row).T)
 
        flag1 += 1 
   
    
    if row.sub_area=="Ljublino": 
        y=model_116.predict_price(pd.DataFrame(row).T)
 
        flag1 += 1 
   
    
    if row.sub_area=="Strogino": 
        y=model_117.predict_price(pd.DataFrame(row).T)
 
        flag1 += 1 
   
    
    if row.sub_area=="Koptevo": 
        y=model_118.predict_price(pd.DataFrame(row).T)
 
        flag1 += 1 
   
    
    if row.sub_area=="Brateevo": 
        y=model_119.predict_price(pd.DataFrame(row).T)
 
        flag1 += 1 
   
    
    if row.sub_area=="Gagarinskoe": 
        y=model_120.predict_price(pd.DataFrame(row).T)
 
        flag1 += 1 
   
    
    if row.sub_area=="Babushkinskoe": 
        y=model_121.predict_price(pd.DataFrame(row).T)
 
        flag1 += 1 
   
    
    if row.sub_area=="Troparevo-Nikulino": 
        y=model_122.predict_price(pd.DataFrame(row).T)
 
        flag1 += 1 
   
    
    if row.sub_area=="Dorogomilovo": 
        y=model_123.predict_price(pd.DataFrame(row).T)
 
        flag1 += 1 
   
    
    if row.sub_area=="Cheremushki": 
        y=model_124.predict_price(pd.DataFrame(row).T)
 
        flag1 += 1 
   
    
    if row.sub_area== "Marfino": 
        y=model_125.predict_price(pd.DataFrame(row).T)
 
        flag1 += 1 
   
    
    if row.sub_area=="Vnukovo": 
        y=model_126.predict_price(pd.DataFrame(row).T)
 
        flag1 += 1 
   
    
    if row.sub_area=="Begovoe": 
        y=model_127.predict_price(pd.DataFrame(row).T)
 
        flag1 += 1 
   
    
    if row.sub_area=="Levoberezhnoe": 
        y=model_128.predict_price(pd.DataFrame(row).T)
 
        flag1 += 1 
   
    
    if row.sub_area=="Prospekt Vernadskogo": 
        y=model_129.predict_price(pd.DataFrame(row).T)
 
        flag1 += 1 
   
    
    if row.sub_area=="Nagatinskij Zaton": 
        y=model_130.predict_price(pd.DataFrame(row).T)
 
        flag1 += 1 
   
    
    if row.sub_area=="Savelki": 
        y=model_131.predict_price(pd.DataFrame(row).T)
 
        flag1 += 1 
   
    
    if row.sub_area=="Sokol'niki": 
        y=model_132.predict_price(pd.DataFrame(row).T)
 
        flag1 += 1 
   
    
    if row.sub_area=="Arbat": 
        y=model_133.predict_price(pd.DataFrame(row).T)
 
        flag1 += 1 
   
    
    if row.sub_area=="Metrogorodok": 
        y=model_134.predict_price(pd.DataFrame(row).T)
 
        flag1 += 1 
   
    
    if row.sub_area=="Poselenie Pervomajskoe": 
        y=model_135.predict_price(pd.DataFrame(row).T)
 
        flag1 += 1 
   
    
    if row.sub_area=="Poselenie Mosrentgen": 
        y=model_136.predict_price(pd.DataFrame(row).T)
 
        flag1 += 1 
   
    
    if row.sub_area== "Poselenie Marushkinskoe": 
        y=model_137.predict_price(pd.DataFrame(row).T)
 
        flag1 += 1 
   
    
    if row.sub_area=="Poselenie Shhapovskoe": 
        y=model_138.predict_price(pd.DataFrame(row).T)
 
        flag1 += 1 
   
    
    if row.sub_area=="Poselenie Rjazanovskoe": 
        y=model_139.predict_price(pd.DataFrame(row).T)
 
        flag1 += 1 
   
    
    if row.sub_area=="Poselenie Kokoshkino": 
        y=model_140.predict_price(pd.DataFrame(row).T)
 
        flag1 += 1 
   
    
    if row.sub_area== "Vostochnoe": 
        y=model_141.predict_price(pd.DataFrame(row).T)
 
        flag1 += 1 
   
    
    if row.sub_area== "Poselenie Voronovskoe": 
        y=model_142.predict_price(pd.DataFrame(row).T)
 
        flag1 += 1 
   
    
    if row.sub_area==  "Poselenie Klenovskoe": 
        y=model_143.predict_price(pd.DataFrame(row).T)
 
        flag1 += 1 
   
    
    if row.sub_area== "Poselenie Kievskij": 
        y=model_144.predict_price(pd.DataFrame(row).T)
 
        flag1 += 1 
   
    
    if row.sub_area==  "Molzhaninovskoe": 
        y=model_145.predict_price(pd.DataFrame(row).T)
 
        flag1 += 1 
   
    
    if row.sub_area==  "Molzhaninovskoe": 
        y=model_145.predict_price(pd.DataFrame(row).T)
 
        flag1 += 1 
   
    
    if row.sub_area== "Poselenie Mihajlovo-Jarcevskoe": 
        y=model_146.predict_price(pd.DataFrame(row).T)
 
        flag1 += 1 
   
    
    if row.sub_area== "Poselenie Rogovskoe": 
        y=model_15.predict_price(pd.DataFrame(row).T)
 
        flag1 += 1 
   
    
    if row.sub_area== "Poselenie Krasnopahorskoe": 
        y=model_18.predict_price(pd.DataFrame(row).T)
 
        flag1 += 1 
    if flag1 != 1:
        print ("\n\n\n oh dear Lord \n\n\n",row)
    predicted.append([row.id,y])

predicted=np.array(predicted)
np.savetxt("final.csv",predicted)




