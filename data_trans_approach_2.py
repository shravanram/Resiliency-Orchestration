#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 15 12:37:16 2021

@author: shravanar
"""
import pandas as pd
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error,mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVC,SVR

scaler = StandardScaler()
df=pd.read_csv("/home/shravanar/Desktop/RO/apprdotmOLTPdb_monitoring_and_rpo_data.csv").drop(columns=["drpol_Deviation","Unnamed: 0","g_name","g_id","drpo_ConfiguredRPO"])
df=df.dropna()
df= pd.DataFrame(scaler.fit_transform(df[["drpol_CurrentRPO",'CPU-Usage (%)',\
                                          'Memory-Usage (%)','Network-Usage Rate (KBps)', 'Disk-Read IOPS', 'Disk-Read Latency (ms)', 'Disk-Write IOPS','Disk-Write Latency (ms)', 'Disk-Read Throughput (KBps)','Disk-Write Throughput (KBps)']]),columns=["drpol_CurrentRPO",'CPU-Usage (%)', 'Memory-Usage (%)','Network-Usage Rate (KBps)', 'Disk-Read IOPS', 'Disk-Read Latency (ms)', 'Disk-Write IOPS','Disk-Write Latency (ms)', 'Disk-Read Throughput (KBps)','Disk-Write Throughput (KBps)'])
n=7

cols=list(df.columns)
#REmove this comment for approach 2
cols.pop(cols.index("drpol_CurrentRPO"))
new_cols=[]
for i in range(n,0,-1):
    #rint(i)
    for col in cols:
        new_cols.append("t"+str(i)+col)


new_cols.append("t1_pred")        
transformed_df=pd.DataFrame(columns=new_cols)



for index in range(0,len(df)): 
    lst=[]
    
    if index+n > len(df)-1:
        break
    for sub_index in range(index,index+n):
        
        for column_name in cols:
            lst.append(df.loc[sub_index,column_name])
    #print(sub_index)
    #remove +1 for approach 2
    lst.append(df.loc[sub_index,"drpol_CurrentRPO"])
        #lst.append(sub_index+1)
        #temp_df["t"+str(count)+column_name]=df.loc[i,column_name]
    temp_df=pd.DataFrame([lst],columns=new_cols)
    transformed_df=pd.concat([transformed_df,temp_df])


training_set, test_set = train_test_split(transformed_df, test_size = 0.3, random_state = 1,shuffle=True)



X_train = training_set.iloc[:,0:len(training_set.columns)-1]
Y_train = training_set.iloc[:,len(training_set.columns)-1]
X_test = test_set.iloc[:,0:len(test_set.columns)-1]
Y_test = test_set.iloc[:,len(training_set.columns)-1]

regr = linear_model.LinearRegression()
regr.fit(X_train, Y_train)    

Y_pred = regr.predict(X_test)
test_set["Predictions"] = Y_pred

r2_score(test_set["t1_pred"],test_set["Predictions"])
mean_absolute_error(test_set["t1_pred"],test_set["Predictions"])
mean_squared_error(test_set["t1_pred"],test_set["Predictions"])
test_set.drop(columns=["Predictions"],inplace=True)
#RF
regressor = RandomForestRegressor(n_estimators = 100, random_state = 0)

regressor.fit(X_train, Y_train)

Y_pred = regressor.predict(X_test)

test_set["Predictions"] = Y_pred

r2_score(test_set["t1_pred"],test_set["Predictions"])
mean_absolute_error(test_set["t1_pred"],test_set["Predictions"])
mean_squared_error(test_set["t1_pred"],test_set["Predictions"])
regressor.feature_importances_

test_set.drop(columns=["Predictions"],inplace=True)

regressor = SVR(kernel='rbf')
regressor.fit(X_train, Y_train)

Y_pred = regressor.predict(X_test)

test_set["Predictions"] = Y_pred

r2_score(test_set["t1_pred"],test_set["Predictions"])
mean_absolute_error(test_set["t1_pred"],test_set["Predictions"])
mean_squared_error(test_set["t1_pred"],test_set["Predictions"])



