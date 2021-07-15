#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 14 11:44:58 2021

@author: shravanar
"""

import pandas as pd
import numpy as np
from sklearn.svm import SVC,SVR
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.metrics import f1_score, accuracy_score,confusion_matrix, plot_confusion_matrix,classification_report
from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error,mean_squared_error




scaler = StandardScaler()
df=pd.read_csv("/home/shravanar/Desktop/RO/apprdotmOLTPdb_monitoring_and_rpo_data.csv")

df=pd.read_csv("/home/shravanar/Desktop/RO/apprdotmOLTPdb_monitoring_and_rpo_data.csv").drop(columns=["Unnamed: 0","index","g_name","g_id","drpo_ConfiguredRPO"])
df=df.dropna()
df_scaled= pd.DataFrame(scaler.fit_transform(df[['CPU-Usage (%)',\
                                          'Memory-Usage (%)','Network-Usage Rate (KBps)', 'Disk-Read IOPS', 'Disk-Read Latency (ms)', 'Disk-Write IOPS','Disk-Write Latency (ms)', 'Disk-Read Throughput (KBps)','Disk-Write Throughput (KBps)']]),columns=['CPU-Usage (%)', 'Memory-Usage (%)','Network-Usage Rate (KBps)', 'Disk-Read IOPS', 'Disk-Read Latency (ms)', 'Disk-Write IOPS','Disk-Write Latency (ms)', 'Disk-Read Throughput (KBps)','Disk-Write Throughput (KBps)'])
df_scaled["drpol_Deviation"]=df["drpol_Deviation"].values


df_scaled.loc[df_scaled["drpol_Deviation"]>0,"drpol_Deviation"]=1


training_set, test_set = train_test_split(df_scaled, test_size = 0.3, random_state = 1)


#Y_test.value_counts()
#Y_train.value_counts()

X_train = training_set.iloc[:,0:len(training_set.columns)-1]
Y_train = training_set.iloc[:,len(training_set.columns)-1]
X_test = test_set.iloc[:,0:len(test_set.columns)-1]
Y_test = test_set.iloc[:,len(training_set.columns)-1]
Y_test.value_counts()
Y_train.value_counts()
classifier = SVC(kernel='rbf', random_state = 1)
classifier.fit(X_train,Y_train)

Y_pred = classifier.predict(X_test)
test_set["Predictions"] = Y_pred
print(classification_report(test_set["drpol_Deviation"],test_set["Predictions"])) 













scaler = StandardScaler()


df=pd.read_csv("/home/shravanar/Desktop/RO/apprdotmOLTPdb_monitoring_and_rpo_data.csv").drop(columns=["drpol_Deviation","Unnamed: 0","index","g_name","g_id","drpo_ConfiguredRPO"])
df=df.dropna()
df_scaled= pd.DataFrame(scaler.fit_transform(df[["drpol_CurrentRPO",'CPU-Usage (%)',\
                                          'Memory-Usage (%)','Network-Usage Rate (KBps)', 'Disk-Read IOPS', 'Disk-Read Latency (ms)', 'Disk-Write IOPS','Disk-Write Latency (ms)', 'Disk-Read Throughput (KBps)','Disk-Write Throughput (KBps)']]),columns=["drpol_CurrentRPO",'CPU-Usage (%)', 'Memory-Usage (%)','Network-Usage Rate (KBps)', 'Disk-Read IOPS', 'Disk-Read Latency (ms)', 'Disk-Write IOPS','Disk-Write Latency (ms)', 'Disk-Read Throughput (KBps)','Disk-Write Throughput (KBps)'])
df_scaled["drpol_CurrentRPO"]=df["drpol_CurrentRPO"].values
data=df_scaled



training_set, test_set = train_test_split(df_scaled, test_size = 0.3, random_state = 1)


Y_test.value_counts()
Y_train.value_counts()

X_train = training_set.iloc[:,0:len(training_set.columns)-1]
Y_train = training_set.iloc[:,len(training_set.columns)-1]
X_test = test_set.iloc[:,0:len(test_set.columns)-1]
Y_test = test_set.iloc[:,len(training_set.columns)-1]


classifier = SVR(kernel='rbf')
classifier.fit(X_train,Y_train)

Y_pred = classifier.predict(X_test)
test_set["Predictions"] = Y_pred

r2_score(Y_test,Y_pred)
mean_absolute_error(Y_test,Y_pred)
mean_squared_error(Y_test,Y_pred)

print(classification_report(test_set["drpol_Deviation"],test_set["Predictions"])) 