import pandas as pd
import numpy as np
import random as rd
import os

data = pd.read_csv("data/201511-final-traindata.csv")
data=np.array(data,dtype='float32')
time  =15
gap=time//5 #15min,30min,60min
temp=0
print(data.shape)
new_flow=[]
group=[]
data_new = np.zeros((int(len(data)/gap),366))
for i in range(len(data)//gap+1):
    data_new[i-1,:] = np.sum(data[(i - 1) * gap:i * gap ],0)
# new_data=np.array(new_data)
print(data_new.shape)
new_data=pd.DataFrame(data_new)
new_data.to_csv('201511-final-traindata_'+str(time)+'min.csv')



