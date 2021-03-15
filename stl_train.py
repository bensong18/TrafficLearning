#STL_cnn+lstm
import torch
from torch.utils.data import Dataset,DataLoader,TensorDataset
import torch.nn as nn
import torchvision.transforms as transforms
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import random
os.environ['PYTHONHASHSEED'] = '0'
seed=14
np.random.seed(seed)
random.seed(seed)
import torch
from torch.autograd import Variable
torch.manual_seed(6)
import time

# Hyper Parameters
INPUT_SIZE = 6      # input size（输入特征维度）
LR = 0.001           # learning rate(Adam-0.01,SGD-0.2)
TIME_STEPS = 6     # time step（步长：使用前TIME_STEP 条数据进行预测）
batch_size = 288
class_cov_num=4
Epoch=200
t=0
s=7

model_path="stlflow2_epo" #训练单任务flow时
# model_path="stlspeed2_epo" #训练单任务speed时

# divide ts as training/testing samples, looBack is lag window(time step)
# NOTE: we can generate the samples as RNN format
def createSamples(dataset, lookBack, RNN=True):
    dataX, dataY = [], []
    for i in range(len(dataset) - lookBack):
        sample_X = dataset[i:(i + lookBack), :]
        sample_Y = dataset[i + lookBack, :]
        dataX.append(sample_X)
        dataY.append(sample_Y)
    dataX = np.array(dataX)  # (N, lag, 1)
    dataY = np.array(dataY)  # (N, 1)
    if not RNN:
        dataX = np.reshape(dataX, (dataX.shape[0], dataX.shape[1]))
    return dataX, dataY

#加载数据
data = pd.read_csv("20151101cba-mtl2.csv")
data = data.drop(['Datetime','speed'], axis = 1) #训练单任务flow时
# data = data.drop(['Datetime','flow'], axis = 1) #训练单任务speed时
data=np.array(data,dtype='float32')
flow=data[:,0] #drop之后的第0列
# speed=data[:,1]
#print(data)
scaler1=MinMaxScaler()
scaler2=MinMaxScaler()
# scaler3=MinMaxScaler()
data=scaler1.fit_transform(data)
flow=flow.reshape(-1,1)
# speed=speed.reshape(-1,1)
flow=scaler2.fit_transform(flow)
# speed=scaler3.fit_transform(speed)
flow_data = data[:,0].reshape(len(data),1)
# speed_data = data[:,1].reshape(len(data),1)
###create dataset
data_X,_ = createSamples(data,TIME_STEPS)
_,data_f = createSamples(flow_data,TIME_STEPS)
# _,data_s = createSamples(speed_data,TIME_STEPS)

length = 396 * 288 -288*153-TIME_STEPS #122对应到7月
print(length,len(data_X))
train_X = data_X[288 * (t + 182):length - 288 * (30 - t+s), :, :]  # 丢弃前s个数据，因总数据不可被batch整除，而drop_last会丢弃后s个数据，影响更大
train_f = data_f[288 * (t + 182):length - 288 * (30 - t+s)]
val_X = data_X[length - 288 * (30 - t+s):length - 288 * (30 - t), :, :]
val_f = data_f[length - 288 * (30 - t+s):length - 288 * (30 - t)]

class Time_Series_Data(Dataset):
    def __init__(self, train_x, train_y1):
        self.X = train_x
        self.y1 = train_y1
        # self.y2 = train_y2
    def __getitem__(self, item):
        x_t = self.X[item]
        y_t1 = self.y1[item]
        return x_t, y_t1
    def __len__(self):
        return len(self.X)

traindataset = Time_Series_Data(train_X,train_f)
trainloader=DataLoader(traindataset,batch_size=batch_size,shuffle=False)#drop_last=True 丢弃最后一个残缺batch
valdataset = Time_Series_Data(val_X, val_f)
valloader = DataLoader(valdataset, batch_size=batch_size, shuffle=False)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=INPUT_SIZE,out_channels=64,kernel_size=1),
            #nn.BatchNorm1d(200),
            nn.ReLU(),
            nn.Dropout(0.5),
            #nn.MaxPool1d(kernel_size=3)
        )
        self.linear = nn.Linear(64,64)
        self.bilstmf = nn.LSTM(
            # input_size=64*TIME_STEPS,
            input_size=64,
            hidden_size=32,
            num_layers=1,
            batch_first=True,
            bidirectional=True #bilstm
        )
        self.dropoutf=nn.Dropout(0.5)
        self.linearf1 = nn.Linear(32*2,class_cov_num) #bilstm应该2*hidden_size
        self.relu = nn.ReLU()
        self.linearf2 = nn.Linear(class_cov_num*TIME_STEPS,1)  # bilstm应该2*hidden_size
        nn.init.xavier_uniform_(self.linearf1.weight,gain=1)
        self.linearf1.bias.data.fill_(0.0)
        nn.init.xavier_uniform_(self.linearf2.weight, gain=1)
        self.linearf2.bias.data.fill_(0.0)
        self.hidden_cellf = None

    def forward(self,x):
        convout = self.conv1(x.transpose(1, 2))
        convout = self.linear(convout.view(len(convout), TIME_STEPS, -1))
        lstm_outf, self.hidden_cellf = self.bilstmf(convout.view(len(convout), TIME_STEPS, -1), self.hidden_cellf)
        lstm_outf = self.dropoutf(lstm_outf)
        fc_outf = self.linearf1(lstm_outf.view(len(lstm_outf), TIME_STEPS, -1))
        fc_outf = self.relu(fc_outf)
        predf = self.linearf2(fc_outf.view(len(convout), -1))
        return predf

model = Net()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(),lr=LR)
# optimizer = torch.optim.RMSprop(model.parameters(),lr=LR)
#optimizer = torch.optim.SGD(model.parameters(),lr=0.1, momentum=0.9, weight_decay=0.0005)
print(model)

#训练参数
time1=time.time()
loss_train,loss_val=[],[]
for epo in range(1,Epoch+1):
    model.train()
    for i, (X, yf) in enumerate(trainloader): # i表示经过了几批batch
        var_x = Variable(X)
        var_yf = Variable(yf)

        optimizer.zero_grad()
        model.hidden_cellf = None
        # forward
        outf = model(var_x)
        # print(outf.shape)
        # print(var_yf.shape)
        loss = criterion(outf,var_yf)

        # backward
        loss.backward()
        optimizer.step()
    loss_train.append(loss)

    model.eval()
    model.hidden_cellf = None
    for j, (X_, yf_) in enumerate(valloader):
        X_ = Variable(X_)
        var_yf_ = Variable(yf_)
        val_f = model(X_)
        loss_ = criterion(val_f, var_yf_)
    loss_val.append(loss_)
    print(f'epoch: {epo:3} loss: {loss.item():10.6f} valloss: {loss_.item():10.6f}')
    if epo>=80 and epo%10==0:
        torch.save(model.state_dict(),"model_result/"+model_path+str(epo)+".pth")

time2=time.time()
print("运行秒数：",time2-time1)

plt.figure()
plt.title('stl train-val loss')
plt.plot(loss_train,'b',label='loss_train')
plt.plot(loss_val,'r',label='loss_val')
plt.legend(loc='upper left')
plt.show()