#两个prior模型的输出,for循环
import torch.optim.lr_scheduler as lr_scheduler
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
import tensor_op
import time

torch.manual_seed(6)
# Hyper Parameters
INPUT_SIZE = 7 # input size（输入特征维度）
TIME_STEPS = 6 # time step（步长：使用前TIME_STEP 条数据进行预测）
batch_size = 288
cov_update_freq=50
class_cov_num=4
# t = 0
# s = 0
#验证时，t设为0，s设为1到s_max；测试时，s设为0，t设为0-29
model_path="model_result/mrnModel3_epo180.pth"

# divide ts as training/testing samples,looBack is lag window(time step)
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
data = data.drop(['Datetime'], axis=1)
data = np.array(data, dtype='float32')
flow = data[:, 0]  # drop之后的第0列
speed = data[:, 1]
scaler1 = MinMaxScaler()
scaler2 = MinMaxScaler()
scaler3 = MinMaxScaler()
data = scaler1.fit_transform(data)
flow = flow.reshape(-1, 1)
speed = speed.reshape(-1, 1)
flow = scaler2.fit_transform(flow)
speed = scaler3.fit_transform(speed)
flow_data = data[:, 0].reshape(len(data), 1)
speed_data = data[:, 1].reshape(len(data), 1)
##create dataset
data_X, _ = createSamples(data, TIME_STEPS)
_, data_f = createSamples(flow_data, TIME_STEPS)
_, data_s = createSamples(speed_data, TIME_STEPS)

length = 396 * 288 -288*153-TIME_STEPS #153对应到6月

class Time_Series_Data(Dataset):
    def __init__(self, train_x, train_y1, train_y2):
        self.X = train_x
        self.y1 = train_y1
        self.y2 = train_y2

    def __getitem__(self, item):
        x_t = self.X[item]
        y_t1 = self.y1[item]
        y_t2 = self.y2[item]
        return x_t, y_t1, y_t2

    def __len__(self):
        return len(self.X)

# testdataset = Time_Series_Data(test_X, test_f, test_s)
# testloader = DataLoader(testdataset, batch_size=batch_size, shuffle=False)

class MRNmodel(nn.Module):
    def __init__(self):
        super(MRNmodel, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=INPUT_SIZE, out_channels=64, kernel_size=1),
            nn.ReLU(),
            nn.Dropout(0.5),
        )
        self.linear=nn.Linear(64,64)
        self.bilstmf = nn.LSTM(
            # input_size=units*TIME_STEPS,
            input_size=64,
            hidden_size=32,
            num_layers=1,
            batch_first=True,
            bidirectional=True  # bilstm
        )
        self.bilstms = nn.LSTM(
            # input_size=64*TIME_STEPS,
            input_size=64,
            hidden_size=32,
            num_layers=1,
            batch_first=True,
            bidirectional=True  # bilstm
        )
        self.dropoutf = nn.Dropout(0.5)
        self.dropouts = nn.Dropout(0.5)
        # self.convf = nn.Sequential(
        #     nn.Conv1d(in_channels=128, out_channels=64, kernel_size=3),
        #     nn.ReLU(),
        #     nn.Dropout(0.5),
        # )
        # self.convs = nn.Sequential(
        #     nn.Conv1d(in_channels=128, out_channels=64, kernel_size=3),
        #     nn.ReLU(),
        #     nn.Dropout(0.5),
        # )
        # self.linear=nn.Linear(128,128)
        # self.linearf = nn.Linear(2*64,128) #bilstm应该2*hidden_size
        # self.linears = nn.Linear(2*64,128)
        self.linearf1 = nn.Linear(32*2,class_cov_num)  # bilstm应该2*hidden_size
        self.linears1 = nn.Linear(32*2,class_cov_num)
        self.reluf1 = nn.ReLU()
        self.relus1 = nn.ReLU()
        self.linearf2 = nn.Linear(class_cov_num*TIME_STEPS,1)
        self.linears2 = nn.Linear(class_cov_num*TIME_STEPS,1)
        nn.init.xavier_uniform_(self.linearf1.weight, gain=1)
        nn.init.xavier_uniform_(self.linears1.weight, gain=1)
        self.linearf1.bias.data.fill_(0.0)
        self.linears1.bias.data.fill_(0.0)
        nn.init.xavier_uniform_(self.linearf2.weight, gain=1)
        nn.init.xavier_uniform_(self.linears2.weight, gain=1)
        self.linearf2.bias.data.fill_(0.0)
        self.linears2.bias.data.fill_(0.0)
        self.hidden_cellf = None
        self.hidden_cells = None
        self.trade_off = 1
        self.bottleneck_size = self.linearf1.in_features
        self.bottleneck_size1 = self.linearf2.in_features
        self.iter_num = 0
        self.cov_update_freq = cov_update_freq

        if self.trade_off > 0:
            # initialize covariance matrix
            self.task_cov = torch.eye(2)  # E(2)
            self.class_cov = torch.eye(class_cov_num)  # E(8)
            self.feature_cov = torch.eye(self.bottleneck_size)  # 生成128*128的单位矩阵E(128)
            self.task_cov_var = Variable(self.task_cov)
            self.class_cov_var = Variable(self.class_cov)
            self.feature_cov_var = Variable(self.feature_cov)
            self.task_cov1 = torch.eye(2)  # E(2)
            self.class_cov1 = torch.eye(1)  # E(1)
            self.feature_cov1 = torch.eye(self.bottleneck_size1)  # 生成128*128的单位矩阵E(128)
            self.task_cov_var1 = Variable(self.task_cov1)
            self.class_cov_var1 = Variable(self.class_cov1)
            self.feature_cov_var1 = Variable(self.feature_cov1)


    def forward(self, x):
        # print(x.size())
        convout = self.conv1(x.transpose(1, 2))
        # print(convout.size())
        convout = self.linear(convout.view(len(convout),TIME_STEPS,-1))
        # print(convout.size())
        lstm_outf, self.hidden_cellf = self.bilstmf(convout.view(len(convout),TIME_STEPS,-1),self.hidden_cellf)
        lstm_outs, self.hidden_cells = self.bilstms(convout.view(len(convout),TIME_STEPS,-1),self.hidden_cells)
        lstm_outf = self.dropoutf(lstm_outf)
        lstm_outs = self.dropouts(lstm_outs)
        # lstm_outf = self.convf(lstm_outf.transpose(1, 2))
        # lstm_outs = self.convs(lstm_outs.transpose(1, 2))
        fc_outf = self.linearf1(lstm_outf.view(len(lstm_outf),TIME_STEPS,-1))
        fc_outs = self.linears1(lstm_outs.view(len(lstm_outf),TIME_STEPS,-1))
        fc_outf = self.reluf1(fc_outf)
        fc_outs = self.relus1(fc_outs)
        predf = self.linearf2(fc_outf.view(len(convout), -1))  # .view(len(convout), -1)
        preds = self.linears2(fc_outs.view(len(convout), -1))
        return predf, preds

def mape(y_true, y_pred):
    n = len(y_true)
    mape = sum(np.abs((y_true - y_pred) / y_true)) / n * 100
    return mape

def se(y_true,y_pred):
    n=len(y_true)
    se=sum(((y_true-y_pred)/y_true-mape(y_true,y_pred))**2)/n
    return se

model = MRNmodel()
model.load_state_dict(torch.load(model_path))
mape_f,mape_s=[],[]

#predict
model.hidden_cellf = None
model.hidden_cells = None
mapef=0
mapes=0
for t in range(0,30):
    test_X = data_X[length - 288 * (30 - t):length - 288 * (29 - t), :, :]
    test_f = data_f[length - 288 * (30 - t):length - 288 * (29 - t)]
    test_s = data_s[length - 288 * (30 - t):length - 288 * (29 - t)]
    testdataset = Time_Series_Data(test_X, test_f, test_s)
    testloader = DataLoader(testdataset, batch_size=batch_size, shuffle=False)
    ##predict,预测
    model.eval()
    # model.hidden_cellf=None
    # model.hidden_cells=None
    pred_f, pred_s = [], []
    for j, (X_, yf_,ys_) in enumerate(testloader):
        model.hidden_cellf = None
        model.hidden_cells = None
        X_ = Variable(X_)
        # print('---------------------------')
        # print(X_)
        predf,preds = model(X_)
        predf,preds = predf.data.numpy(),preds.data.numpy()
        predf,preds = np.squeeze(predf),np.squeeze(preds)
        pred_f.append(predf)
        pred_s.append(preds)
    pred_f,pred_s=np.array(pred_f),np.array(pred_s)
    pred_f,pred_s=pred_f.reshape(288,1),pred_s.reshape(288,1)
    pred_f = scaler2.inverse_transform(pred_f)
    test_f = scaler2.inverse_transform(test_f)
    pred_s = scaler3.inverse_transform(pred_s)
    test_s = scaler3.inverse_transform(test_s)
    mapef+=mape(test_f[72:264], pred_f[72:264])
    mapes+=mape(test_s[72:264], pred_s[72:264])

    mape_f.append(mape(test_f[72:264], pred_f[72:264]))
    mape_s.append(mape(test_s[72:264], pred_s[72:264]))
    print('flow MAPE:', mape(test_f[72:264], pred_f[72:264]))  # test_f[72:264], pred_f[72:264]
    print('speed MAPE:', mape(test_s[72:264], pred_s[72:264]))  # test_s[72:264], pred_s[72:264]
    # if t==7:
    #     fig1 = plt.figure()
    #     ax1 = fig1.add_subplot(111)
    #     plt.title('traffic flow prediction')
    #     plt.plot(test_f, 'b', label='real flow')
    #     plt.plot(pred_f, 'r', label='predict flow')
    #     plt.legend(loc='upper left')
    #     xticks = [0, 48, 96, 144, 192, 240, 288]
    #     xlabels = ['0:00', '4:00', '8:00', '12:00', '16:00', '20:00', '24:00']
    #     ax1.set_xticks(xticks)
    #     ax1.set_xticklabels(xlabels)  # , rotation='45') #rotation='vertical'
    #     plt.xlim(0, 288)
    #     plt.show()
print("Average flow mape:",mapef/30," Average speed mape:",mapes/30)
# print("Average speed mape:",mapes/30)

mape_f=pd.DataFrame(mape_f)
mape_s=pd.DataFrame(mape_s)
mape_f.to_csv("model_result/mape_f.csv")
mape_s.to_csv("model_result/mape_s.csv")

# print('pred_f[:10]', pred_f[:10], 'pred_f[-10:]', pred_f[-10:])
# print('test_f[:10]', test_f[:10], 'test_f[-10:]', test_f[-10:])
# print('pred_s[:10]', pred_s[:10], 'pred_s[-10:]', pred_s[-10:])


# fig1 = plt.figure()
# ax1 = fig1.add_subplot(111)
# plt.title('traffic flow prediction')
# plt.plot(test_f, 'b', label='real flow')
# plt.plot(pred_f, 'r', label='predict flow')
# plt.legend(loc='upper left')
# xticks = [0, 48, 96, 144, 192, 240, 288]
# xlabels = ['0:00', '4:00', '8:00', '12:00', '16:00', '20:00', '24:00']
# ax1.set_xticks(xticks)
# ax1.set_xticklabels(xlabels)  # , rotation='45') #rotation='vertical'
# plt.xlim(0, 288)
# plt.ylim(0, 530)
#
# # plt.figure()
# # plt.title('difference flow prediction')
# # plt.plot(test_f[0:287], 'b', label='real flow')
# # plt.plot(pred_f[1:288], 'r', label='predict flow')
# # plt.legend(loc='upper left')
#
# fig2 = plt.figure()
# ax2 = fig2.add_subplot(111)
# plt.title('traffic speed prediction')
# plt.plot(test_s, 'b', label='real speed')
# plt.plot(pred_s, 'r', label='predict speed')
# plt.legend(loc='upper left')
# ax2.set_xticks(xticks)
# ax2.set_xticklabels(xlabels)
# plt.xlim(0, 288)
# plt.ylim(min(test_s) - 20, max(test_s + 20))
# plt.show()