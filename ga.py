# -*- coding: utf-8 -*-
"""
Created on Wed Sep  9 09:32:35 2020

@author: Administrator
"""


import numpy as np
from sko.GA import GA
import pandas as pd
import matplotlib.pyplot as plt
import time

data = pd.read_csv("C:\\Users\\Administrator\\Desktop\\参数求解\\1.csv")
data = data.iloc[:5,:]
t = data.iloc[:,0]
h = data.iloc[:,1]
waterin = data.iloc[:,2]
flow = data.iloc[:,3]
hz = data.iloc[:,4]
waterout = data.iloc[:,5]
    
def CoolingTower(dryball,humidity,inletwater,waterflow,Hz,outletwater,c,n,a):
    PVSain=10**(2.7877+7.625*dryball/(241.6+dryball))
    PVain=PVSain*humidity/100
    Hain=1.006*dryball+(0.621955*PVain)/(101325-PVain)*(2501+1.83*dryball)
    
    PVSswi=10**(2.7877+7.625*inletwater/(241.6+inletwater))*1 #adjustcoef
    Hswi=1.006*inletwater+(0.621955*PVSswi)/(101325-PVSswi)*(2501+1.83*inletwater)
    
    outletwater=inletwater-5
    PVSswo=10**(2.7877+7.625*outletwater/(241.6+outletwater))
    Hswo=1.006*outletwater+(0.621955*PVSswo)/(101325-PVSswo)*(2501+1.83*outletwater)
    
    Cs=(Hswi-Hswo)/(inletwater-outletwater)
    
    ma=a*Hz/50 #to know 730 is a good value #455不同
    ma=ma/(0.0000136*36**2-0.0046675*36+1.29295)*(0.0000136*(inletwater-3)**2-0.0046675*(inletwater-3)+1.29295) #修正
    # ma adjusted coef, depend on density of air
    mw=waterflow
    Cpw=4.1868
    mstar=ma/mw*Cs/Cpw
    
    if mstar>1:
        mstar=1/mstar
    
    NTU=c*(ma/mw)**(-1-n)
    # adjusted coef for too much air
    if ma/mw*0.95 > 273/385 :
        adj_air_wt=np.exp(-ma/mw*385/273*0.092)
    else :
        adj_air_wt=1
    sigmaa=(1-np.exp(-NTU*(1-mstar)))/(1-mstar*np.exp(-NTU*(1-mstar)))*adj_air_wt
    Qrej=sigmaa*ma*(Hswi-Hain)
    return abs(inletwater-Qrej/waterflow/4.1868-outletwater)

#%%
def objectfunction(p):
    x1, x2 ,x3 = p
    res = [0 for i in range(len(t))]
    for i in range(len(t)):
        res[i] = CoolingTower(t[i],h[i],waterin[i],flow[i],hz[i],waterout[i],x1,x2,x3)
    return sum(res)

#%%
time_start=time.time()
ga = GA(func=objectfunction, n_dim=3, size_pop=100, max_iter=1000, lb=[-10, -10, -10], ub=[10, 10, 10], precision=1e-7)
best_x, best_y = ga.run()
print('best_x:', best_x, '\n', 'best_y:', best_y)
time_end=time.time()
print('time cost of GA',time_end-time_start,'s', '-'*70)


Y_history = pd.DataFrame(ga.all_history_Y)
fig, ax = plt.subplots(2, 1)
ax[0].plot(Y_history.index, Y_history.values, '.', color='red')
Y_history.min(axis=1).cummin().plot(kind='line')
plt.show()