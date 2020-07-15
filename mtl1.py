from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Dense, Input
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.optimizers import Adam
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' #ignore

inputs = Input(shape=(6,))
x = Dense(576,activation='relu')(inputs)
x = Dense(576, activation='relu')(x)
x = Dense(576, activation='relu')(x)
x = Dense(576, activation='relu')(x)
x = Dense(576, activation='relu')(x)
sub1=Dense(288, activation='relu')(x)
#sub1=Dense(288,activation='relu')(sub1)
sub2=Dense(288, activation='relu')(x)
#sub2=Dense(288,activation='relu')(sub2)
output1 = Dense(
        units=1,
        name="output1"
    )(sub1)
output2 = Dense(
        units=1,
        name="output2"
    )(sub2)

model = Model(
        inputs=inputs,
        outputs=[output1, output2],
        name="my_model"
    )
Adam(lr=0.002)
model.compile(
    optimizer='adam',
    loss={'output1': 'mse', 'output2':'mse'},
    loss_weights={'output1': 1.0, 'output2':0.2},
    metrics=['mape']
    )
#加载数据
#在Excel进行了数据归一化预处理
data=pd.read_csv('20151109-20161109c.csv')
X=data.iloc[:,1:7] # feature,第7列上游车流不加入训练
y1=data.iloc[:,0] # label speed
y2=data.iloc[:,8] # label flow
X,y1,y2=np.array(X).astype('float32'),np.array(y1).astype('float32'),np.array(y2).astype('float32')
length=len(X)
t=length-288*7
train_X,train_y1,train_y2=X[0:t],y1[0:t],y2[0:t]#training data
Xtest,y1test,y2test=X[t:],y1[t:],y2[t:]#全体测试集（7天）
test_X,test_y1,test_y2=Xtest[288*0:288*1],y1test[288*0:288*1],y2test[288*0:288*1]#测试集第1天

model.fit(train_X,[train_y1,train_y2],epochs=30,batch_size=576,validation_split=0.05,verbose=2)
pred_y=model.predict(test_X,batch_size=144,steps=1)

pred_y=np.array(pred_y,dtype=np.float32)#shape:(2,288,1),2array,288行1列
pred_y=pred_y.reshape(576,)
pred_speed=pred_y[0:288]
pred_flow=pred_y[288:]
print(pred_y)
def mape(y_true, y_pred):
    n = len(y_true)
    mape = sum(np.abs((y_true - y_pred) / y_true)) / n * 100
    return mape
print('speed MAPE:',mape(test_y1[72:264],pred_speed[72:264]))
print('flow MAPE:',mape(test_y2[72:264],pred_flow[72:264]))
def se(y_true,y_pred):
    n=len(y_true)
    se=sum(((y_true-y_pred)/y_true-mape(y_true,y_pred))**2)/n
    return se
print('speed SE:',se(test_y1[72:264],pred_speed[72:264]))
print('flow SE:',se(test_y2[72:264],pred_flow[72:264]))

plt.figure(1)
plt.title('MTL traffic speed prediction')
plt.plot(test_y1 ,'b',label='real speed')
plt.plot(pred_speed,'r', label='predict speed')
plt.legend(loc='upper left')
plt.figure(2)
plt.title('MTL traffic flow prediction')
plt.plot(test_y2 ,'b',label='real flow')
plt.plot(pred_flow,'r',label='predict flow')
plt.legend(loc='upper left')
plt.show()

