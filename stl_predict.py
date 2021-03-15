#加入一个for循环，一次性输出测试集n天预测结果，并保存到excel中
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch.nn as nn
import torchvision.transforms as transforms
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import random
os.environ['PYTHONHASHSEED'] = '0'
seed = 14
np.random.seed(seed)
random.seed(seed)
import torch
from torch.autograd import Variable
torch.manual_seed(6)

# Hyper Parameters
INPUT_SIZE = 6      # input size（输入特征维度）
LR = 0.001           # learning rate(Adam-0.01,SGD-0.2)
TIME_STEPS = 6     # time step（步长：使用前TIME_STEP 条数据进行预测）
batch_size = 288
units=80
class_cov_num=4
# s = 0#To output the MAPE and SE values of the validation set,modify s.(s=1->10/31)Set it to 0 when testing.
# model_path="model_result/singlef_MaxEpoch2.pth" #注意是f还是s
model_path="model_result/stlspeed2_epo200.pth" #注意是f还是s


# divide ts as training/testing samples, looBack is lag window(time step)
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

# 加载数据
data = pd.read_csv("20151101cba-mtl2.csv")
# data = data.drop(['Datetime', 'speed'], axis=1)  # when predict flow
data = data.drop(['Datetime','flow'], axis = 1) #when predict speed
data = np.array(data, dtype='float32')
flow = data[:, 0]  # drop之后的第0列
# print(data)
scaler1 = MinMaxScaler()
scaler2 = MinMaxScaler()
data = scaler1.fit_transform(data)
flow = flow.reshape(-1, 1)
flow = scaler2.fit_transform(flow)
flow_data = data[:, 0].reshape(len(data), 1)
###create dataset
data_X, _ = createSamples(data, TIME_STEPS)
_, data_f = createSamples(flow_data, TIME_STEPS)
length = 396 * 288 -288*153-TIME_STEPS #122对应到7月

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

def mape(y_true, y_pred):
    n = len(y_true)
    mape = sum(np.abs((y_true - y_pred) / y_true)) / n * 100
    return mape

def se(y_true,y_pred):
    n=len(y_true)
    se=sum(((y_true-y_pred)/y_true-mape(y_true,y_pred))**2)/n
    return se

model = Net()
model.load_state_dict(torch.load(model_path))
mape_sum=[]
mapef=0
for t in range(0,30):
    test_X = data_X[length - 288 * (30 - t):length - 288 * (29 - t), :, :]
    test_f = data_f[length - 288 * (30 - t):length - 288 * (29 - t)]
    testdataset = Time_Series_Data(test_X, test_f)
    testloader = DataLoader(testdataset, batch_size=batch_size, shuffle=False)
    ##predict,预测
    model.eval()
    # model.hidden_cellf=None
    # model.hidden_cells=None
    pred_f=[]
    for j, (X_, yf_) in enumerate(testloader):
        model.hidden_cellf = None
        X_ = Variable(X_)
        # print('---------------------------')
        # print(X_)
        predf = model(X_)
        predf = predf.data.numpy()
        predf = np.squeeze(predf)
        pred_f.append(predf)
    pred_f=np.array(pred_f)
    pred_f=pred_f.reshape(288,1)
    pred_f=scaler2.inverse_transform(pred_f)
    test_f=scaler2.inverse_transform(test_f)
    mapef += mape(test_f[72:264], pred_f[72:264])

    mape_sum.append(mape(test_f[72:264], pred_f[72:264]))
    print('MAPE:', mape(test_f[72:264], pred_f[72:264]))
    # print('SE:', se(test_f[72:264], pred_f[72:264]))
print("Average mape:",mapef/30)

mape_sum=pd.DataFrame(mape_sum)
mape_sum.to_csv("model_result/mape_sum.csv")

# plt.figure()
# plt.title('traffic flow/speed prediction')
# plt.plot(test_f,'b',label='real')
# plt.plot(pred_f,'r', label='predict')
# plt.legend(loc='upper left')
# plt.show()




