from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
from statsmodels.tsa.exponential_smoothing.ets import ETSModel

t = 8 #预测的该天在总数据中倒数第几天
s = 4 #预测的该天在对应组中倒数第几天
group=3 #预测的这天属于哪组

# 加载数据
data = pd.read_csv("data/201511-final-traindata.csv")
# data = data.drop(['datetime'],axis = 1)
data = np.array(data,dtype='float32')
# print(data,data.shape)
# flow_data = data[:,0]
scaler1 = MinMaxScaler()
# scaler2 = MinMaxScaler()
flow = data[1:].reshape(-1, 1)
test_f = data[1:,366-t].reshape(-1,1)
test_fe = test_f
flow = scaler1.fit_transform(flow)

##全年数据预测，训练参数
all_data=[]
pred_y=[]
# coef=[]
for i in range(288):#288
    for j in range(366):
        all_data.append(data[i+1,j]/1010)

    all_data = pd.Series(all_data)
    model = ETSModel(all_data[0:366 - t], error='add', trend='add', damped_trend=True)
    model_fit = model.fit(maxiter=1000, disp=0)

    output=model_fit.forecast()
    # print(output)
    pred=output
    # print(pred)
    pred_y.append(pred)
    all_data = []
# print(pred_y)
pred_f=np.array(pred_y)
pred_f=pred_f.reshape(288,1)
pred_f=scaler1.inverse_transform(pred_f)
# print(pred_f,test_f)

#分组预测，训练参数
all_data1=[]
pred_y1=[]
for i in range(288):#288
    for j in range(366):
        if data[0,j]==group:
            all_data1.append(data[i+1,j]/1010)

    all_data1 = pd.Series(all_data1)
    model1 = ETSModel(all_data1[0:366 - t], error='add', trend='add', damped_trend=True)
    model_fit1 = model1.fit(maxiter=1000, disp=0)

    output1=model_fit1.forecast()
    pred1=output1
    pred_y1.append(pred1)
    all_data1 = []
# print(pred_y)
pred_f1=np.array(pred_y1)
pred_f1=pred_f1.reshape(288,1)
pred_f1=scaler1.inverse_transform(pred_f1)
# print(pred_f1,test_f)

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

def mase(true,pred):
    true = np.array(true)
    pred = np.array(pred)
    return np.mean(np.abs(pred-true))/np.mean(np.abs(true[1:]-true[:-1]))

print('72:264_MAPE:',mape(test_f[72:264],pred_f[72:264]))
print('nomatch MAPE:',mape(test_fe,pred_f))#晚高峰[198:234]
print('72:264_SE:',se(test_f[72:264],pred_f[72:264]))
print('nomatch SE:',se(test_fe,pred_f))
print('72:264_MASE:',mase(test_f[72:264],pred_f[72:264]))
print('nomatch MASE:',mase(test_fe,pred_f))
print('72:264_SMAPE:',smape(test_f[72:264],pred_f[72:264]))
print('nomatch SMAPE:',smape(test_fe,pred_f))
print('-------------------')
print('72:264_MAPE:',mape(test_f[72:264],pred_f1[72:264]))
print('match MAPE:',mape(test_fe,pred_f1))
print('72:264_SE:',se(test_f[72:264],pred_f1[72:264]))
print('match SE:',se(test_fe,pred_f1))
print('72:264_MASE:',mase(test_f[72:264],pred_f1[72:264]))
print('match MASE:',mase(test_fe,pred_f1))
print('72:264_SMAPE:',smape(test_f[72:264],pred_f1[72:264]))
print('match SMAPE:',smape(test_fe,pred_f1))

pred_save=pd.DataFrame(pred_f)
pred_save.to_csv("pred_oneday.csv")
pred_save1=pd.DataFrame(pred_f1)
pred_save1.to_csv("pred_oneday1.csv")

fig1=plt.figure()
# ax1 = fig1.add_subplot(111)
plt.title('traffic flow prediction')
plt.plot(test_fe,'b',label='real')
plt.plot(pred_f,'r', label='predict')
plt.legend(loc='upper left')

fig2=plt.figure()
# ax2 = fig2.add_subplot(111)
plt.title('traffic flow prediction')
plt.plot(test_fe,'b',label='real')
plt.plot(pred_f1,'r', label='predict')
plt.legend(loc='upper left')

plt.show()