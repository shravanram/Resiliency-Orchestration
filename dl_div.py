#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 20 13:27:05 2021

@author: shravanar
"""
import sklearn
from sklearn import preprocessing
#!/usr/bin/env python
# coding: utf-8
data_df = pd.read_csv(path).interpolate().drop(columns="Unnamed: 0")
data_df=data_df.loc[data_df["g_name"]=="RG_apprdotmOLTPdb"]

# In[115]:


import pandas as pd
import os

from keras.models import *
from keras.layers import *
from keras.layers.core import Lambda
from keras import backend as K
from sklearn import preprocessing
from keras.callbacks import EarlyStopping
from pickle import dump
import sklearn
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler,StandardScaler

import matplotlib.pyplot as plt


# In[116]:

path="/home/shravanar/Desktop/RO/22dec to 10may_asian_paints_7hosts_monitoring_and_rpo_data.csv"
data_path = "AP-Perf-data/"
files = os.listdir(data_path)
hosts = [i  for i in files if ".csv" in i]


# In[117]:


host = "apprdotmOLTPdb_monitoring_and_rpo_data.csv"

data_df = pd.read_csv(path).interpolate().drop(columns="Unnamed: 0")
data_df=data_df.loc[data_df["g_name"]=="RG_apprdotmOLTPdb"]

'''
scaler = StandardScaler()
scaled_df = pd.DataFrame(scaler.fit_transform(data_df[["drpol_Deviation","drpol_CurrentRPO",'CPU-Usage (%)', 'Memory-Usage (%)','Network-Usage Rate (KBps)', 'Disk-Read IOPS', 'Disk-Read Latency (ms)','Disk-Total IOPS', 'Disk-Total Latency (ms)','Disk-Total Throughput (KBps)', 'Disk-Write IOPS','Disk-Write Latency (ms)', 'Disk-Read Throughput (KBps)','Disk-Write Throughput (KBps)']]),columns=["drpol_Deviation","drpol_CurrentRPO",'CPU-Usage (%)', 'Memory-Usage (%)','Network-Usage Rate (KBps)', 'Disk-Read IOPS', 'Disk-Read Latency (ms)','Disk-Total IOPS', 'Disk-Total Latency (ms)','Disk-Total Throughput (KBps)', 'Disk-Write IOPS','Disk-Write Latency (ms)', 'Disk-Read Throughput (KBps)','Disk-Write Throughput (KBps)'])

# columns = ['drpo_ConfiguredRPO','drpol_CurrentRPO', 'drpol_Deviation', 'CPU-Usage (%)','Memory-Usage (%)', 'Network-Usage Rate (KBps)', 'Disk-Read IOPS','Disk-Read Latency (ms)', 'Disk-Write IOPS','Disk-Write Latency (ms)', 'Disk-Read Throughput (KBps)','Disk-Write Throughput (KBps)']
columns = ["drpol_Deviation","drpol_CurrentRPO",'CPU-Usage (%)', 'Memory-Usage (%)','Network-Usage Rate (KBps)', 'Disk-Read IOPS', 'Disk-Read Latency (ms)','Disk-Total IOPS', 'Disk-Total Latency (ms)','Disk-Total Throughput (KBps)', 'Disk-Write IOPS','Disk-Write Latency (ms)', 'Disk-Read Throughput (KBps)','Disk-Write Throughput (KBps)']
features = scaled_df[columns[2:]].values
labels = scaled_df[columns[0]].values

features = data_df[columns[2:]].values
labels = data_df[columns[0]].values
'''

scaler1 = StandardScaler()
scaler2 = StandardScaler()

scaled_df_x = pd.DataFrame(scaler1.fit_transform(data_df[['CPU-Usage (%)', 'Memory-Usage (%)','Network-Usage Rate (KBps)', 'Disk-Read IOPS', 'Disk-Read Latency (ms)','Disk-Total IOPS', 'Disk-Total Latency (ms)','Disk-Total Throughput (KBps)', 'Disk-Write IOPS','Disk-Write Latency (ms)', 'Disk-Read Throughput (KBps)','Disk-Write Throughput (KBps)']]),columns=['CPU-Usage (%)', 'Memory-Usage (%)','Network-Usage Rate (KBps)', 'Disk-Read IOPS', 'Disk-Read Latency (ms)','Disk-Total IOPS', 'Disk-Total Latency (ms)','Disk-Total Throughput (KBps)', 'Disk-Write IOPS','Disk-Write Latency (ms)', 'Disk-Read Throughput (KBps)','Disk-Write Throughput (KBps)'])
scaled_df_y = pd.DataFrame(scaler2.fit_transform(data_df[["drpol_Deviation"]]),columns=["drpol_Deviation"])

features = scaled_df_x.values
labels = scaled_df_y.values


# In[119]:


# data_df["DateTime"] = pd.to_datetime(data_df['index'],format="%Y-%m-%d %H:%M")


# In[120]:


seq_length = 24
X,y = [],[]
#need to test with y = i+seq_length-1
for i in range(len(features)-seq_length-1):
    X.append(features[i:(i+seq_length)])
    y.append(float(labels[i+seq_length]))

X = np.array(X)
y = np.array(y)


# In[121]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train = np.array(X_train)
X_test = np.array(X_test)
y_train = np.array(y_train)
y_test = np.array(y_test)


# In[122]:


def q_loss(q,y,f):
    e = (y-f)
    return K.mean(K.maximum(q*e, (q-1)*e), axis=-1)

losses = [lambda y,f: q_loss(0.1,y,f), lambda y,f: q_loss(0.5,y,f), lambda y,f: q_loss(0.9,y,f)]
inputs = Input(shape=(X_train.shape[1], X_train.shape[2]))
lstm = Bidirectional(LSTM(64, return_sequences=True, dropout=0.1))(inputs, training = True)
lstm = Bidirectional(LSTM(16, return_sequences=False, dropout=0.1))(lstm, training = True)
dense = Dense(50)(lstm)
out10 = Dense(1)(dense)
out50 = Dense(1)(dense)
out90 = Dense(1)(dense)
model = Model(inputs, [out10,out50,out90])
model.compile(loss=losses, optimizer='adam', loss_weights = [0.3,0.3,0.3])


# In[123]:


model.fit(X_train, [y_train,y_train,y_train], epochs=10, batch_size=256,
                    validation_data = (X_test,[y_test,y_test,y_test]),
                    callbacks=[EarlyStopping(monitor='val_loss', patience=5)],verbose=2, shuffle=True)


# In[124]:


ls = model.predict(X_test)


'''
import matplotlib.pyplot as plt
start,end = 100,200
# plt.plot(ls[0][start:end],label="Lower Baseline")
plt.plot(ls[1][start:end],label="Predicted Value")
# plt.plot(ls[2][start:end],label="Upper Baseline")
plt.plot(y_test[start:end],label="Actual Value")
plt.rcParams["figure.figsize"] = (15,15)
plt.legend()
plt.show()
'''





# In[103]:


relevant_val = [i for i in range(len(y_test)) if y_test[i]>0]
# relevant_val


# In[87]:


relevant_train = [i for i in range(len(y_train)) if y_train[i]>0]
# relevant_train

