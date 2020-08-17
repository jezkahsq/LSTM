# -*- coding: utf-8 -*-
"""
Created on Tue Aug 11 10:18:06 2020

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

data_num_start = 24000 #从第data_num_start行数据开始训练
data_num_end = 28000 #训练结束行数
data_num = data_num_end - data_num_start #读取训练样本数
val_num = 110 #输入的预测样本个数 实际上预测的是最后val_num-timesteps个 val_num应大于等于timesteps
epoch_num = 300
timesteps = 10
neuron_num = 50
data = pd.read_excel('C:\\Users\\Administrator\\Desktop\\多变量时间序列\\滑动窗口筛选1.xlsx')
df = np.array(data.iloc[data_num_start: data_num_end])
x, y =[], []
for i in range(len(df) - timesteps):
    end_index = i + timesteps
    if end_index > len(df):
        break
    else:
        seq_x = df[i:end_index, :] #读了所有列，包括目标预测列
        seq_y = df[end_index-1, -1]
        x.append(seq_x)
        y.append(seq_y)
        
n_features = np.array(x).shape[2]
x = np.array(x)
y = np.array(y)
y = utils.to_categorical(y, num_classes=3)
model = Sequential()
model.add(LSTM(neuron_num, input_shape = (timesteps, n_features)))
model.add(Dropout(0.6))
model.add(Dense(3, activation='softmax'))
model.add(Activation('sigmoid'))
model.compile(optimizer ='adam', loss = 'categorical_crossentropy', metrics=['accuracy'])
model.fit(x, y, epochs = epoch_num, verbose = 1)
#joblib.dump(model, 'C:\\Users\\Administrator\\Desktop\\多变量时间序列\\LSTM')


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
        seq_y = xy_input[end_index-1, -1]
        x_val.append(seq_x)
        y_val.append(seq_y)
        
n_features = np.array(x_val).shape[2]
x_val = np.array(x_val)
y_val = np.array(y_val)

x_input = x_val.reshape((len(x_val), timesteps, n_features))  #转换成样本量+步长+特征的格式
y_pre= model.predict_classes(x_input)

        

j = 0
k = []
for i in range(len(y_val)):
    if y_val[i] != y_pre[i]:
        j += 1
        k.append(i + 1)
print(f'训练数据区间{(data_num_start, data_num_end)}', '\n',
      f'模型参数(epoch, timesteps, neuron)={epoch_num, timesteps, neuron_num}', '\n', 
      '预测趋势错误的时间点个数:', j, '\n',
      '预测错误的位置:', k, '\n')


'''
yval = np.concatenate((y, y_val), axis = 0)
y_pre = y_pre.reshape((1, len(y_pre)))
ypre = np.concatenate((y, y_pre[0]), axis = 0)

ypre_trend = np.array([])
for i in range(1,len(ypre)):
    c = ypre[i] - ypre[i-1]
    ypre_trend = np.append(ypre_trend, c)

yval_trend = np.array([])
for i in range(1,len(yval)):
    d = yval[i] - yval[i-1]
    yval_trend = np.append(yval_trend, d)


diff_ = np.array([])
for i in range(len(yval_trend)):
    if yval_trend[i] < 0:
        if ypre_trend[i] >= 0:
            diff_2 = abs(yval_trend[i] - ypre_trend[i]) * abs(yval_trend[i] - ypre_trend[i])
            diff_ = np.append(diff_, diff_2)
        else:
            diff_ = np.append(diff_, 0) 
    
    elif yval_trend[i] > 0:
        if ypre_trend[i] <= 0:
            diff_1 = abs(yval_trend[i] - ypre_trend[i]) * abs(yval_trend[i] - ypre_trend[i])
            diff_ = np.append(diff_, diff_1)
        else:
            diff_ = np.append(diff_, 0)
            
    else:
        if ypre_trend[i] != 0:
            diff_3 = abs(yval_trend[i] - ypre_trend[i]) * abs(yval_trend[i] - ypre_trend[i])
            diff_ = np.append(diff_, diff_3)
        else:
            diff_ = np.append(diff_, 0)
diff_sum = np.sum(diff_)
print(f'训练数据区间{(data_num_start, data_num_end)}', '\n',
      f'模型参数(epoch, timesteps, neuron)={epoch_num, timesteps, neuron_num}', '\n', 
      '趋势判断错误率:', diff_sum, '\n',
      '预测时间点个数:', val_num-timesteps, '\n')

j = 0
k = []
error_index_interval = data_num - timesteps - 2
for i in range(len(diff_)):
    if diff_[i] != 0:
       j += 1
       k.append(i - error_index_interval)
print('预测趋势错误的时间点个数:', j, '\n',
      '预测错误的位置:', k)
     
    
dt1 = {'x':[], 'y':[]}
for x in range(len(yval)):
    dt1['x'].append(x)
    dt1['y'].append(yval[x])
plot_dt1 = pd.DataFrame(dt1, columns = ['x', 'y'])

dt2 = {'x':[], 'y':[]}
for x in range(len(ypre)):
    dt2['x'].append(x)
    dt2['y'].append(ypre[x])
plot_dt2 = pd.DataFrame(dt2, columns = ['x', 'y'])

plot_num = data_num - timesteps 
plt.figure(figsize=(20,6))
l1 ,= plt.plot(plot_dt1.x[:plot_num], plot_dt1.y[:plot_num], linewidth = 3)
l2 ,= plt.plot(plot_dt1.x[plot_num - 1:], plot_dt1.y[plot_num - 1:], linewidth = 2)    
l3 ,= plt.plot(plot_dt2.x[plot_num - 1:], plot_dt2.y[plot_num - 1:], linewidth = 2) 
plt.legend([l1,l2,l3],('raw-data','true-values','pre-values'), loc='best')
plt.show()
'''