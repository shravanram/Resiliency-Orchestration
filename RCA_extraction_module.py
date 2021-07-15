#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  3 11:34:47 2021

@author: shravanar
"""
                #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 20 13:27:05 2021

@author: shravanar
"""

# In[115]:

#!/usr/bin/env python
# coding: utf-8

# In[115]:

'''
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
'''
import pandas as pd
import os
import tensorflow
tensorflow.compat.v1.disable_v2_behavior()
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
#from tensorflow.keras.layers.core import Lambda
#import tensorflow.keras.layers.Lambda
from tensorflow.keras import backend as K
from sklearn import preprocessing
from tensorflow.keras.callbacks import EarlyStopping
from pickle import dump
from tensorflow.keras.layers import Activation, Dense

from pickle import dump
import sklearn
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler,StandardScaler

# In[116]:

path="/home/shravanar/Desktop/RO/22dec to 10may_asian_paints_7hosts_monitoring_and_rpo_data.csv"
data_path = "AP-Perf-data/"
files = os.listdir(data_path)
hosts = [i  for i in files if ".csv" in i]


# In[117]:


host = "apprdotmOLTPdb_monitoring_and_rpo_data.csv"

data_df = pd.read_csv(path).interpolate().drop(columns="Unnamed: 0")
data_df=data_df.loc[data_df["g_name"]=="RG_apprdotmOLTPdb"].reset_index(drop=True)

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

scaled_df_x = pd.DataFrame(scaler1.fit_transform(data_df[['CPU-Usage (%)', 'Memory-Usage (%)','Network-Usage Rate (KBps)', 'Disk-Read IOPS', 'Disk-Read Latency (ms)', 'Disk-Write IOPS','Disk-Write Latency (ms)', 'Disk-Read Throughput (KBps)','Disk-Write Throughput (KBps)']]),columns=['CPU-Usage (%)', 'Memory-Usage (%)','Network-Usage Rate (KBps)', 'Disk-Read IOPS', 'Disk-Read Latency (ms)', 'Disk-Write IOPS','Disk-Write Latency (ms)', 'Disk-Read Throughput (KBps)','Disk-Write Throughput (KBps)'])
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

split=int(len(X)*0.8)

X_train=X[0:split]
X_test=X[split:]
y_train=y[0:split]
y_test=y[split:]

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


model.fit(X_train, [y_train,y_train,y_train], epochs=5, batch_size=256,
                    validation_data = (X_test,[y_test,y_test,y_test]),
                    callbacks=[EarlyStopping(monitor='val_loss', patience=5)],verbose=2, shuffle=True)
print(model.summary())

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




relevant_train = [i for i in range(len(y_train)) if y_train[i]>0]
# relevant_train

                           
#df=pd.read_csv("/home/shravanar/Desktop/RO/codingcsv/shap_df.csv").drop(columns="Unnamed: 0")
#df["min"]=df.min(axis=1)

#df_list=df.values.tolist()

#%%


import shap  # package used to calculate Shap values

# Create object that can calculate shap values
explainer = shap.DeepExplainer(model,X_train[:10])


#X_test[0][0].shape


shap_values = explainer.shap_values(X_test[560:600,:],check_additivity=False)
shap.initjs()
#model.predict(X_test[0:101])
#explainer = shap.KernelExplainer(model.predict, X_test)


num_timestamps=len(shap_values[0][0])-1

total_pred=len(shap_values[0])
largest_influence_df=pd.DataFrame(columns=scaled_df_x.columns)
for i in range(total_pred):
    largest_influence_df=largest_influence_df.append(pd.DataFrame([shap_values[0][i][num_timestamps]], columns=scaled_df_x.columns))

df_list=largest_influence_df.values.tolist()
names=largest_influence_df.columns


res=[]
for _list in df_list:
    print
    sorted_list=sorted(_list)
    direction=[0,1,-1,-2]
    temp=[]
    for i in direction:
        temp.append(names[_list.index(sorted_list[i])])
    res.append(temp)
    
result=pd.DataFrame(res)
result.columns=["min1","min2","max1","max2"]



merged_copy=pd.concat([data_df[split+560:split+600].reset_index(drop=True),result],axis=1)

merged_copy["avg Disk-Write Throughput (KBps)"]=data_df["Disk-Write Throughput (KBps)"].mean()

merged_copy["avg Disk-Read Throughput (KBps)"]=data_df["Disk-Read Throughput (KBps)"].mean()
merged_copy["avg Memory-Usage (%)"]=data_df["Memory-Usage (%)"].mean()

merged_copy["avg Disk-Write Latency (ms)"]=data_df["Disk-Write Latency (ms)"].mean()


#merged_df=pd.concat([df,dummy],axis=1)
shap_values

X_test[0].shape

