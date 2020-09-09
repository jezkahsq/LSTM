# -*- coding: utf-8 -*-
"""
Created on Tue Sep  8 14:54:45 2020

@author: Administrator
"""


import numpy as np
from scipy import optimize
import time




t = [24,23,22]
h = [70,100,80]
waterin = [300,300,400]
flow = [50,50,40]
hz = [100,500,300]
waterout = [50,40,50]
    
def CoolingTower(dryball,humidity,inletwater,waterflow,Hz,outletwater,c,n):
    PVSain=10**(2.7877+7.625*dryball/(241.6+dryball))
    PVain=PVSain*humidity/100
    Hain=1.006*dryball+(0.621955*PVain)/(101325-PVain)*(2501+1.83*dryball)
    
    PVSswi=10**(2.7877+7.625*inletwater/(241.6+inletwater))*1 #adjustcoef
    Hswi=1.006*inletwater+(0.621955*PVSswi)/(101325-PVSswi)*(2501+1.83*inletwater)
    
    outletwater=inletwater-5
    PVSswo=10**(2.7877+7.625*outletwater/(241.6+outletwater))
    Hswo=1.006*outletwater+(0.621955*PVSswo)/(101325-PVSswo)*(2501+1.83*outletwater)
    
    Cs=(Hswi-Hswo)/(inletwater-outletwater)
    
    ma=455*Hz/50 #to know 730 is a good value #455不同
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
    return inletwater-Qrej/waterflow/4.1868

def bisect_(t,h,waterin,flow,hz,waterout,N):
    
    coolingfanHz1=optimize.bisect(lambda x: CoolingTower(t,h,waterin,flow,hz,waterout,x,N)-waterout,0.0001,1000) #,xtol=0.001,rtol=0.01
    print(coolingfanHz1)
    
def newton_(t,h,waterin,flow,hz,waterout):

    coolingfanHz2=optimize.newton(lambda x,y: helper(x,y),[0,0])
    print(coolingfanHz2)
    
def helper(x, y):
    res = [0 for i in range(len(t))]
    for i in range(len(t)):
        res[i] = CoolingTower(t[i],h[i],waterin[i],flow[i],hz[i],waterout[i],x,y)
    return res
    
time_start=time.time()
newton_(t,h,waterin,flow,hz,waterout)
time_end=time.time()
print('time cost of newton',time_end-time_start,'s', '\n')
# time_start=time.time()
# for i in range(len(t)):
#     newton_(t[i],h[i],waterin[i],flow[i],hz[i],waterout[i])
# time_end=time.time()
# print('time cost of newton',time_end-time_start,'s')
