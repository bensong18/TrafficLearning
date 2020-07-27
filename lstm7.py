# cnn+bilstm+(attension)+mtl
from keras.layers import Input, Dense, LSTM, merge ,Conv1D,Dropout,Bidirectional,Multiply
from keras.models import Model
from attention_utils import get_activations
from keras.layers import merge
from keras.layers.core import *
from keras.layers.recurrent import LSTM
from keras.models import *
from sklearn.preprocessing import MinMaxScaler
import  pandas as pd
import  numpy as np
import matplotlib.pyplot as plt


SINGLE_ATTENTION_VECTOR = False
def attention_3d_block(inputs):
    # inputs.shape = (batch_size, time_steps, input_dim)
    input_dim = int(inputs.shape[2])
    a = inputs
    #a = Permute((2, 1))(inputs)
    #a = Reshape((input_dim, TIME_STEPS))(a) # this line is not useful. It's just to know which dimension is what.
    a = Dense(input_dim, activation='softmax')(a)
    if SINGLE_ATTENTION_VECTOR:
        a = Lambda(lambda x: K.mean(x, axis=1), name='dim_reduction')(a)
        a = RepeatVector(input_dim)(a)
    a_probs = Permute((1, 2), name='attention_vec')(a)

    output_attention_mul = merge([inputs, a_probs], name='attention_mul', mode='mul')
    return output_attention_mul
# 注意力机制的另一种写法 适合上述报错使用 来源:https://blog.csdn.net/uhauha2929/article/details/80733255
def attention_3d_block2(inputs, single_attention_vector=False):
    # 如果上一层是LSTM，需要return_sequences=True
    # inputs.shape = (batch_size, time_steps, input_dim)
    time_steps = K.int_shape(inputs)[1]
    input_dim = K.int_shape(inputs)[2]
    a = Permute((2, 1))(inputs)
    a = Dense(time_steps, activation='softmax')(a)
    if single_attention_vector:
        a = Lambda(lambda x: K.mean(x, axis=1))(a)
        a = RepeatVector(input_dim)(a)
    a_probs = Permute((2, 1))(a)
    # 乘上了attention权重，但是并没有求和，好像影响不大
    # 如果分类任务，进行Flatten展开就可以了
    # element-wise
    output_attention_mul = Multiply()([inputs, a_probs])
    return output_attention_mul


def create_dataset(dataset, look_back):
    '''
    对数据进行处理
    '''
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back),:]
        dataX.append(a)
        dataY.append(dataset[i + look_back,:])

    dataX = np.array(dataX)
    dataY = np.array(dataY)
    return dataX, dataY

def attention_model():
    inputs = Input(shape=(TIME_STEPS, INPUT_DIMS),name='inputs')

    x = Conv1D(filters = 64, kernel_size = 1, activation = 'relu')(inputs)  #, padding = 'same'
    x = Dropout(0.3)(x)

    #lstm_out = Bidirectional(LSTM(lstm_units, activation='relu'), name='bilstm')(x)
    #对于GPU可以使用CuDNNLSTM
    lstm_outf = Bidirectional(LSTM(lstm_units, return_sequences=True))(x)
    lstm_outs = Bidirectional(LSTM(lstm_units, return_sequences=True))(x)

    lstm_outf = Dropout(0.3)(lstm_outf) ## 含attention
    lstm_outs = Dropout(0.3)(lstm_outs)
    attention_mulf = attention_3d_block2(lstm_outf)
    attention_mulf = Flatten()(attention_mulf)
    attention_muls = attention_3d_block2(lstm_outs)
    attention_muls = Flatten()(attention_muls)
    outf = Dense(1,name='outf')(attention_mulf)#activation=sigmoid
    outs = Dense(1, name='outs')(attention_muls)

    # lstm_outf=Flatten()(lstm_outf) ## 不含attention
    # lstm_outs=Flatten()(lstm_outs)
    # outf = Dense(1, name='outf')(lstm_outf)
    # outs = Dense(1, name='outs')(lstm_outs)

    model = Model(
        inputs=inputs,
        outputs=[outf, outs],
        name="my_model"
    )
    return model

INPUT_DIMS = 6
TIME_STEPS = 20
lstm_units = 64


#加载数据
data = pd.read_csv("20151109lstm.csv")
data = data.drop(['Datetime'], axis = 1)
data=np.array(data)
flow=data[:,0]
speed=data[:,1]
#print(data)
scaler1=MinMaxScaler()
scaler2=MinMaxScaler()
scaler3=MinMaxScaler()
data=scaler1.fit_transform(data)
flow,speed=flow.reshape(-1,1),speed.reshape(-1,1)
flow=scaler2.fit_transform(flow)
speed=scaler3.fit_transform(speed)
flow_data = data[:,0].reshape(len(data),1) #drop之后的第0列
speed_data = data[:,1].reshape(len(data),1)
###create dataset
data_X,_ = create_dataset(data,TIME_STEPS)
_,data_f = create_dataset(flow_data,TIME_STEPS)
_,data_s = create_dataset(speed_data,TIME_STEPS)

length=len(data_X)
t=4
train_X=data_X[288*t:length-288*(30-t),:,:] # predict 11.1-11.30
train_f=data_f[288*t:length-288*(30-t)]
train_s=data_s[288*t:length-288*(30-t)]
test_X=data_X[length-288*(30-t):length-288*(29-t),:,:]
test_f=data_f[length-288*(30-t):length-288*(29-t)]
test_s=data_s[length-288*(30-t):length-288*(29-t)]
#print(test_f,test_s)

m = attention_model()
m.summary()
m.compile(
    optimizer='adam',
    loss={'outf': 'mse', 'outs':'mse'},
    loss_weights={'outf': 10, 'outs':0.2},
    metrics=['mape']
    )#
m.fit(train_X, [train_f,train_s], epochs=1, batch_size=64, validation_split=0.1,verbose=2,shuffle=False)
pred_y=m.predict(test_X)
pred_y=np.array(pred_y,dtype=np.float32)#shape:(2,288,1),2array,288行1列
pred_y=pred_y.reshape(576,)
pred_f=pred_y[:288]
pred_s=pred_y[288:]
#print(pred_f,pred_s)
pred_f=pred_f.reshape(-1,1)
pred_s=pred_s.reshape(-1,1)
pred_f=scaler2.inverse_transform(pred_f)
pred_s=scaler3.inverse_transform(pred_s)
test_f=scaler2.inverse_transform(test_f)
test_s=scaler3.inverse_transform(test_s)
#print(pred_f,pred_s)
#print(test_f,test_s)

def mape(y_true, y_pred):
    n = len(y_true)
    mape = sum(np.abs((y_true - y_pred) / y_true)) / n * 100
    return mape
print('flow MAPE:',mape(test_f[72:264],pred_f[72:264]))
print('speed MAPE:',mape(test_s[72:264],pred_s[72:264]))

def se(y_true,y_pred):
    n=len(y_true)
    se=sum(((y_true-y_pred)/y_true-mape(y_true,y_pred))**2)/n
    return se
print('flow SE:',se(test_f[72:264],pred_f[72:264]))
print('speed SE:',se(test_s[72:264],pred_s[72:264]))

plt.figure()
plt.title('traffic flow prediction')
plt.plot(test_f,'b',label='real flow')
plt.plot(pred_f,'r', label='predict flow')
plt.legend(loc='upper left')
plt.figure()
plt.title('traffic speed prediction')
plt.plot(test_s,'b',label='real speed')
plt.plot(pred_s,'r', label='predict speed')
plt.legend(loc='upper left')
plt.show()

from keras.utils import plot_model
plot_model(m, to_file='model1.png')