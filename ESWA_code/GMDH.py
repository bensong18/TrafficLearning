import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
from keras.layers import Input, Dense, LSTM, merge ,Conv1D,Dropout
from keras.models import Model,Sequential
from keras.layers.normalization import BatchNormalization as bn
from keras.layers import merge
from keras.layers.core import *
from keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import random
import tensorflow as tf
from gmdhpy.gmdh import Regressor
os.environ['PYTHONHASHSEED'] = '0'
seed=7
np.random.seed(seed)
tf.compat.v1.set_random_seed(seed)
random.seed(seed)

TIME_STEPS = 4# time step（步长：使用前TIME_STEP 条数据进行预测）
LR = 0.001 #default 0.001

time = 5
gap=time//5 #15min/5min,30min/5min,...

Epoch = 50 #
t =8##总数据的倒数第几天
s =4#分组预测时用，表示要预测该组的倒数第几天
group=3 #预测天属于哪组

def createSamples(dataset, lookBack):
    dataX, dataY = [], []
    for i in range(len(dataset) - lookBack):
        sample_X = dataset[i:(i + lookBack), :]
        sample_Y = dataset[i + lookBack, :]
        dataX.append(sample_X)
        dataY.append(sample_Y)
    dataX = np.array(dataX)  # (N, lag, 1)
    dataY = np.array(dataY)  # (N, 1)
    return dataX,dataY

# 加载数据
data = pd.read_csv("data/201511-final-traindata.csv")#_'+str(time)+'min
# data = data.drop(['datetime'],axis = 1)
data = np.array(data,dtype='float32')
# print(data)
# flow_data = data[:,0]
# print(flow_data)
# print(flow_data.shape)#(105408,)
scaler1 = MinMaxScaler()
# scaler2 = MinMaxScaler()
flow = data[1:].reshape(-1, 1)
# print(flow)
# print(flow.shape)#(105408, 1)
test_f = data[1:,366-t].reshape(-1,1)
flow = scaler1.fit_transform(flow)

model = Regressor(ref_functions=('linear_cov','quadratic'))
model1 = Regressor(ref_functions=('linear_cov','quadratic'))

##全年数据预测，训练参数
pred_y=[]
for i in range(288//gap):
    train_data=data[i+1]
    # print(train_data)#[array([0.10382514], dtype=float32), array([0.1147541], dtype=float32),...])
    train_data=train_data.reshape(-1,1)
    train_data=train_data/1010
    # train_data=(train_data-263)/(10666-263)
    # print(train_data)
    # print(train_data.shape)#(368,1)
    data_X,data_y = createSamples(train_data,TIME_STEPS)
    # print(data_X.shape,data_y.shape)
    # print(data_y)
    train_X = data_X[0:len(data_y) - t, :]  # length=366-TS,366-TS-8->11-3
    train_y = data_y[0:len(data_y) - t, :]
    test_X = data_X[len(data_y) - t:len(data_y) - t + 1, :]
    train_X,train_y,test_X=train_X.reshape(len(train_X),4),train_y.reshape(len(train_y),1),test_X.reshape(len(test_X),4)
    # print(train_X.shape,train_y.shape)
    model.fit(train_X,train_y,verbose=0)
    pred=model.predict(test_X)
    pred_y.append(pred)
    train_data = []
# print(pred_y)
pred_f=np.array(pred_y)
pred_f=pred_f.reshape(288//gap,1)
pred_f=scaler1.inverse_transform(pred_f)
# print(pred_f,test_f)

##分组预测，训练参数
train_data1=[]
pred_y1=[]
for i in range(288//gap):#288
    for j in range(366):
        if data[0,j]==group: #只加入属于该组的数据
            train_data1.append(data[i+1,j])
    # print(train_data1)#[array([0.10382514], dtype=float32), array([0.1147541], dtype=float32),...])
    train_data1=np.array(train_data1).reshape(-1,1)
    # print(train_data1)
    train_data1=train_data1/1010
    # print(train_data1)
    data_X1,data_y1 = createSamples(train_data1,TIME_STEPS)
    # print(data_X.shape,data_y.shape)
    # print(data_y)
    # print(len(train_data))
    # print(len(data_y))
    train_X1 = data_X1[0:len(data_y1) - s, :]  #-t->t days before the target day
    train_y1 = data_y1[0:len(data_y1) - s, :]
    test_X1 = data_X1[len(data_y1) - s:len(data_y1) - s + 1, :]
    # test_y1 = data_y1[len(data_y1) - s:len(data_y1) - s + 1, :]
    # print(test_y)
    train_X1, train_y1, test_X1 = train_X1.reshape(len(train_X1), 4), train_y1.reshape(len(train_y1), 1), test_X1.reshape(len(test_X1), 4)
    model1.fit(train_X1,train_y1,verbose=0)
    pred1=model1.predict(test_X1)
    pred_y1.append(pred1)
    train_data1 = []
# print(pred_y)
pred_f1=np.array(pred_y1)
pred_f1=pred_f1.reshape(288//gap,1)
pred_f1=scaler1.inverse_transform(pred_f1)
# print(pred_f,test_f)

def mape(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return np.mean(np.abs((y_pred - y_true) / y_true)) * 100

def se(y_true,y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return np.mean(((2*np.abs(y_true-y_pred)/(y_true+y_pred))/y_true-smape(y_true,y_pred))**2)

def smape(y_true,y_pred):
    y_true=np.array(y_true)
    y_pred=np.array(y_pred)
    return 2.0 * np.mean(np.abs(y_pred-y_true) / (np.abs(y_pred)+np.abs(y_true))) * 100

def mase(true,pred):#缩放mae
    true = np.array(true)
    pred = np.array(pred)
    return np.mean(np.abs(pred-true))/np.mean(np.abs(true[1:]-true[:-1]))

print('72:264_MAPE:',mape(test_f[72//gap:264//gap],pred_f[72//gap:264//gap]))
print('0:288_MAPE:',mape(test_f,pred_f))
print('72:264_SE:',se(test_f[72//gap:264//gap],pred_f[72//gap:264//gap]))
print('0:288_SE:',se(test_f,pred_f))
print('72:264_MASE:',mase(test_f[72//gap:264//gap],pred_f[72//gap:264//gap]))
print('0:288_MASE:',mase(test_f,pred_f))
print('72:264_SMAPE:',smape(test_f[72//gap:264//gap],pred_f[72//gap:264//gap]))
print('0:288_SMAPE:',smape(test_f,pred_f))
print('-------------------')
print('72:264_MAPE:',mape(test_f[72//gap:264//gap],pred_f1[72//gap:264//gap]))
print('0:288_MAPE:',mape(test_f,pred_f1))
print('72:264_SE:',se(test_f[72//gap:264//gap],pred_f1[72//gap:264//gap]))
print('0:288_SE:',se(test_f,pred_f1))
print('72:264_MASE:',mase(test_f[72//gap:264//gap],pred_f1[72//gap:264//gap]))
print('0:288_MASE:',mase(test_f,pred_f1))
print('72:264_SMAPE:',smape(test_f[72//gap:264//gap],pred_f1[72//gap:264//gap]))
print('0:288_SMAPE:',smape(test_f,pred_f1))

fig1=plt.figure()
# ax1 = fig1.add_subplot(111)
plt.title('traffic flow prediction')
plt.plot(test_f,'b',label='real')
plt.plot(pred_f,'r', label='predict')
plt.legend(loc='upper left')

fig2=plt.figure()
# ax2 = fig2.add_subplot(111)
plt.title('traffic flow prediction')
plt.plot(test_f,'b',label='real')
plt.plot(pred_f1,'r', label='predict')
plt.legend(loc='upper left')

plt.show()