#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 20 12:16:22 2021

@author: shravanar
"""

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
  
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error,mean_squared_error
scaler = StandardScaler()


df=pd.read_csv("/home/shravanar/Desktop/RO/apprdotmOLTPdb_monitoring_and_rpo_data.csv").drop(columns=["drpol_Deviation","Unnamed: 0","index","g_name","g_id","drpo_ConfiguredRPO"])
df=df.dropna()
df= pd.DataFrame(scaler.fit_transform(df[["drpol_CurrentRPO",'CPU-Usage (%)',\
                                          'Memory-Usage (%)','Network-Usage Rate (KBps)', 'Disk-Read IOPS', 'Disk-Read Latency (ms)','Disk-Total IOPS', 'Disk-Total Latency (ms)','Disk-Total Throughput (KBps)', 'Disk-Write IOPS','Disk-Write Latency (ms)', 'Disk-Read Throughput (KBps)','Disk-Write Throughput (KBps)']]),columns=["drpol_CurrentRPO",'CPU-Usage (%)', 'Memory-Usage (%)','Network-Usage Rate (KBps)', 'Disk-Read IOPS', 'Disk-Read Latency (ms)','Disk-Total IOPS', 'Disk-Total Latency (ms)','Disk-Total Throughput (KBps)', 'Disk-Write IOPS','Disk-Write Latency (ms)', 'Disk-Read Throughput (KBps)','Disk-Write Throughput (KBps)'])


X = df.drop(columns="drpol_CurrentRPO")
y = df["drpol_CurrentRPO"]

x_train, x_test, y_train, y_test = train_test_split(X, y,random_state=1)
 # create regressor object
regressor = RandomForestRegressor(n_estimators = 100, random_state = 0)

regressor.fit(x_train, y_train)

Y_pred = regressor.predict(x_test)


r2_score(Y_pred,y_test)
mean_absolute_error(Y_pred,y_test)
mean_squared_error(Y_pred,y_test)

