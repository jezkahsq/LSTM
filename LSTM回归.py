# -*- coding: utf-8 -*-
"""
Created on Mon Aug 17 16:11:42 2020

@author: Administrator
"""


import numpy as np
import pandas as pd
from keras import utils
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense, Activation
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import math
from sklearn.externals import joblib
from keras.layers import Dropout
import warnings
warnings.filterwarnings("ignore") 

data_num_start = 0 #从第data_num_start行数据开始训练
data_num_end = 1000 #训练结束行数
data_num = data_num_end - data_num_start #读取训练样本数
val_num = 110 #输入的预测样本个数 实际上预测的是最后val_num-timesteps个 val_num应大于等于timesteps
epoch_num = 200
timesteps = 10
neuron_num = 100
data = pd.read_excel('C:\\Users\\Administrator\\Desktop\\多变量时间序列\\多变量时间序列.xlsx')
df = np.array(data.iloc[data_num_start: data_num_end])
x, y =[], []
for i in range(len(df) - timesteps):
    end_index = i + timesteps
    if end_index > len(df):
        break
    else:
        seq_x = df[i:end_index, :] #读了所有列，包括目标预测列
        seq_y = df[end_index, -1]
        x.append(seq_x)
        y.append(seq_y)
        
n_features = np.array(x).shape[2]
x = np.array(x)
y = np.array(y)

model = Sequential()
model.add(LSTM(neuron_num, activation='relu', input_shape = (timesteps, n_features)))
model.add(Dense(1))
model.compile(optimizer ='adam', loss = 'mse', metrics=['accuracy'])
model.fit(x, y, epochs = epoch_num, verbose = 0)

val_start = data_num_end - timesteps 
val_end = data_num_end + val_num - timesteps 
xy_input = np.array(data.iloc[val_start: val_end])
x_val, y_val =[], []
for i in range(len(xy_input)-timesteps):
    end_index = i + timesteps
    if end_index > len(xy_input):
        break
    else:
        seq_x = xy_input[i:end_index, :]
        seq_y = xy_input[end_index, -1]
        x_val.append(seq_x)
        y_val.append(seq_y)
n_features = np.array(x_val).shape[2]
x_val = np.array(x_val)
y_val = np.array(y_val)
x_input = x_val.reshape((len(x_val), timesteps, n_features))  # 转换成样本量+步长+特征的格式
y_pre= model.predict(x_input)
testScore = math.sqrt(mean_absolute_error(y_val, y_pre))

j = 0
k = []
for i in range(len(y_val)):
    if y_val[i] > 0:
        if y_pre[i] <= 0:
            j += 1
            k.append(i+1)
    elif y_val[i] == 0:
        if y_pre[i] != 0:
        # if y_pre[i] <= -0.005 and y_pre[i] >= 0.005:
            j += 1
            k.append(i+1)
    else:
        if y_pre[i] >= 0:
            j += 1
            k.append(i+1)
            
print(f'训练数据区间{(data_num_start, data_num_end)}', '\n',
      f'模型参数(epoch, timesteps, neuron)={epoch_num, timesteps, neuron_num}', '\n', 
      '预测趋势错误的时间点个数:', j, '\n',
      '预测错误的位置:', k, '\n',
      'Test Score: %.2f MAE' % (testScore))