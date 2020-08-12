# -*- coding: utf-8 -*-
"""
Created on Tue Aug 11 10:18:06 2020

@author: Administrator
"""

import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import math
from sklearn.externals import joblib

data = pd.read_excel('C:\\Users\\Administrator\\Desktop\\多变量时间序列\\多变量时间序列.xlsx')
timesteps = 20
df = np.array(data.iloc[0:1000])
x, y =[], []
for i in range(len(df)):
    end_index = i + timesteps
    if end_index > len(df):
        break
    else:
        seq_x, seq_y = df[i:end_index, :-1], df[end_index - 1, -1]
        x.append(seq_x)
        y.append(seq_y)
        
n_features = np.array(x).shape[2]
x = np.array(x)
y = np.array(y)
model = Sequential()
model.add(LSTM(50, activation='relu', input_shape = (timesteps, n_features)))
model.add(Dense(1))
model.compile(optimizer ='adam', loss = 'mse')
model.fit(x, y, epochs = 400, verbose = 0)
#joblib.dump(model, 'C:\\Users\\Administrator\\Desktop\\多变量时间序列\\LSTM')


xy_input = np.array(data.iloc[1000:1200])
x_val, y_val =[], []
for i in range(len(xy_input)):
    end_index = i + timesteps
    if end_index > len(xy_input):
        break
    else:
        seq_x, seq_y = xy_input[i:end_index, :-1], xy_input[end_index - 1, -1]
        x_val.append(seq_x)
        y_val.append(seq_y)
        
n_features = np.array(x).shape[2]
x_val = np.array(x_val)
y_val = np.array(y_val)

#x_input = x_input.reshape((1231, timesteps, n_features))  # 转换成样本量+步长+特征的格式
y_pre= model.predict(x_val)
testScore = math.sqrt(mean_squared_error(y_val, y_pre))
print('Test Score: %.2f RMSE' % (testScore))


yval = np.concatenate((y, y_val), axis = 0)
y_pre = y_pre.reshape((1, 181))
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
print("趋势判断正确率:", diff_sum, '\n')

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

plt.figure(figsize=(20,6))
l1 ,= plt.plot(plot_dt1.x[:992],plot_dt1.y[:992], linewidth = 3)
l2 ,= plt.plot(plot_dt1.x[991:], plot_dt1.y[991:], linewidth = 2)    
l3 ,= plt.plot(plot_dt2.x[991:], plot_dt2.y[991:], linewidth = 2) 
plt.legend([l1,l2,l3],('raw-data','true-values','pre-values'),loc='best')
plt.show()