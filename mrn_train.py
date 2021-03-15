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
import torch
from torch.autograd import Variable
import tensor_op
import time

torch.manual_seed(6)
# Hyper Parameters
INPUT_SIZE = 7  # input size（输入特征维度）
TIME_STEPS = 6  # time step（步长：使用前TIME_STEP 条数据进行预测）
batch_size = 288
LR = 0.001 #default 0.001
trade_off_value = 1e1
cov_update_freq=50
class_cov_num=4
loss_wf = 1
loss_ws = 2
t = 0 #Set to 0 for training phase
s = 7 #16年11月1日往前推移s天为验证集
Epoch = 200
# device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
model_path="mrnModel3_epo"

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
print(length, len(data_X))
train_X = data_X[288 * (t + 182):length - 288 * (30 - t+s), :, :]#训练阶段，t为0
train_f = data_f[288 * (t + 182):length - 288 * (30 - t+s)]
train_s = data_s[288 * (t + 182):length - 288 * (30 - t+s)]
val_X = data_X[length - 288 * (30 - t+s):length - 288 * (30 - t), :, :]
val_f = data_f[length - 288 * (30 - t+s):length - 288 * (30 - t)]
val_s = data_s[length - 288 * (30 - t+s):length - 288 * (30 - t)]
print(train_X.shape, train_f.shape, val_X.shape, val_f.shape)

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

traindataset = Time_Series_Data(train_X, train_f, train_s)
trainloader = DataLoader(traindataset, batch_size=batch_size, shuffle=False)  # drop_last=True 丢弃最后一个残缺batch
valdataset = Time_Series_Data(val_X, val_f, val_s)
valloader = DataLoader(valdataset, batch_size=batch_size, shuffle=False)

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


model = MRNmodel()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(),LR,weight_decay=0)  # learning rate(Adam-0.001,SGD-0.2),weight-decay=0.0005
# optimizer = torch.optim.SGD(model.parameters(),lr=0.1, momentum=0.9, weight_decay=0.0005)
# optimizer = torch.optim.RMSprop(model.parameters())
# scheduler=lr_scheduler.ReduceLROnPlateau(optimizer,'min',factor=0.5,patience=3)#过程中调低学习率
# scheduler=lr_scheduler.MultiStepLR(optimizer, milestones=[60,90,120,150,180], gamma=0.2)
print(model)

def mape(y_true, y_pred):
    n = len(y_true)
    mape = sum(np.abs((y_true - y_pred) / y_true)) / n * 100
    return mape

# 训练参数
time1=time.time()
loss_train_total,loss_ts,loss_val_total=[],[],[]
mapef6_22,mapes6_22=[],[]
for epo in range(1,Epoch+1):
    model.train()
    for i, (X, yf, ys) in enumerate(trainloader):  # i表示经过了几批batch
        var_x = Variable(X)
        var_yf = Variable(yf)
        var_ys = Variable(ys)

        optimizer.zero_grad()
        model.hidden_cellf = None
        model.hidden_cells = None
        # forward
        outf, outs = model(var_x)
        # print(outf)
        # print(var_yf)
        lossf = criterion(outf, var_yf)
        losss = criterion(outs, var_ys)
        tsloss = lossf * loss_wf + losss * loss_ws
        if model.trade_off > 0:
            weight_size = model.linearf1.weight.size()
            all_weights = [model.linearf1.weight.view(1, weight_size[0], weight_size[1]),
                           model.linears1.weight.view(1, weight_size[0], weight_size[1])]
            weights = torch.cat(all_weights, dim=0).contiguous()
            multi_task_loss = tensor_op.MultiTaskLoss(weights, model.task_cov_var, model.class_cov_var,
                                                      model.feature_cov_var)
            weight_size1 = model.linearf2.weight.size()
            all_weights1 = [model.linearf2.weight.view(1, weight_size1[0], weight_size1[1]),
                           model.linears2.weight.view(1, weight_size1[0], weight_size1[1])]
            weights1 = torch.cat(all_weights1, dim=0).contiguous()
            multi_task_loss1 = tensor_op.MultiTaskLoss(weights1, model.task_cov_var1, model.class_cov_var1,
                                                      model.feature_cov_var1)
            total_loss = tsloss + trade_off_value * (multi_task_loss+multi_task_loss1)  # 各任务的mseloss+trade_off*mtl_loss
            # print(tsloss) #tensor(0.3352, grad_fn=<AddBackward0>)
            # print(multi_task_loss) #tensor([0.0224], grad_fn=<ViewBackward>)
            # print(total_loss) #tensor([0.3576], grad_fn=<AddBackward0>)...
        else:
            total_loss = tsloss

        # backward
        # model.hidden_cellf = None
        # model.hidden_cells = None
        # model.hidden_cellf = (torch.zeros(2, batch_size, 64), #(num_layers * num_directions, batch_size, hidden_size)
        #                       torch.zeros(2, batch_size, 64))
        # model.hidden_cells = (torch.zeros(2, batch_size, 64),  # (num_layers * num_directions, batch_size, hidden_size)
        #                       torch.zeros(2, batch_size, 64))
        model.iter_num += 1 #记录训练的iteration迭代次数，不同于epoch
        total_loss.backward()
        optimizer.step()
        # scheduler.step()

        if model.trade_off > 0 and model.iter_num % model.cov_update_freq == 0:
            # get updated weights #在每个cov_update_freq时更新参数
            weight_size = model.linearf1.weight.size()
            all_weights = [model.linearf1.weight.view(1, weight_size[0], weight_size[1]),
                           model.linears1.weight.view(1, weight_size[0], weight_size[1])]
            weights = torch.cat(all_weights, dim=0).contiguous()
            weight_size1 = model.linearf2.weight.size()
            all_weights1 = [model.linearf2.weight.view(1, weight_size1[0], weight_size1[1]),
                            model.linears2.weight.view(1, weight_size1[0], weight_size1[1])]
            weights1 = torch.cat(all_weights1, dim=0).contiguous()

            ## update cov parameters
            temp_task_cov_var = tensor_op.UpdateCov(weights.data, model.class_cov_var.data, model.feature_cov_var.data)
            temp_class_cov_var = tensor_op.UpdateCov(weights.data.permute(1, 0, 2).contiguous(), model.task_cov_var.data,
                                                     model.feature_cov_var.data)
            temp_feature_cov_var = tensor_op.UpdateCov(weights.data.permute(2, 0, 1).contiguous(),model.task_cov_var.data,
                                                       model.class_cov_var.data)
            temp_task_cov_var1 = tensor_op.UpdateCov(weights1.data, model.class_cov_var1.data, model.feature_cov_var1.data)
            temp_class_cov_var1 = tensor_op.UpdateCov(weights1.data.permute(1, 0, 2).contiguous(),model.task_cov_var1.data,model.feature_cov_var1.data)
            temp_feature_cov_var1 = tensor_op.UpdateCov(weights1.data.permute(2, 0, 1).contiguous(),model.task_cov_var1.data,model.class_cov_var1.data)

            ## task covariance
            u, s, v = torch.svd(temp_task_cov_var)
            # s = s.cpu().apply_(self.select_func)
            model.task_cov_var = torch.mm(u, torch.mm(torch.diag(s), torch.t(v)))  # s的对角线元素，v的转置
            this_trace = torch.trace(model.task_cov_var)
            if this_trace > 3000.0:  # 3000
                model.task_cov_var = Variable(model.task_cov_var / this_trace * 3000.0)
            else:
                model.task_cov_var = Variable(model.task_cov_var)
            u1, s1, v1 = torch.svd(temp_task_cov_var1)
            # s = s.cpu().apply_(self.select_func)
            model.task_cov_var1 = torch.mm(u1, torch.mm(torch.diag(s1), torch.t(v1)))  # s的对角线元素，v的转置
            this_trace1 = torch.trace(model.task_cov_var1)
            if this_trace1 > 3000.0:  # 3000
                model.task_cov_var1 = Variable(model.task_cov_var1 / this_trace1 * 3000.0)
            else:
                model.task_cov_var1 = Variable(model.task_cov_var1)

            ##class covariance
            u, s, v = torch.svd(temp_class_cov_var)
            model.class_cov_var = torch.mm(u, torch.mm(torch.diag(s), torch.t(v)))
            this_trace = torch.trace(model.class_cov_var)
            if this_trace > 3000.0:
                model.class_cov_var = Variable(model.class_cov_var / this_trace * 3000.0)
            else:
                model.class_cov_var = Variable(model.class_cov_var)
            u1, s1, v1 = torch.svd(temp_class_cov_var1)
            # s = s.cpu().apply_(self.select_func)
            model.class_cov_var1 = torch.mm(u1, torch.mm(torch.diag(s1), torch.t(v1)))  # s的对角线元素，v的转置
            this_trace1 = torch.trace(model.class_cov_var1)
            if this_trace1 > 3000.0:  # 3000
                model.class_cov_var1 = Variable(model.class_cov_var1 / this_trace1 * 3000.0)
            else:
                model.class_cov_var1 = Variable(model.class_cov_var1)

            ## feature covariance
            u, s, v = torch.svd(temp_feature_cov_var)
            temp_feature_cov_var = torch.mm(u, torch.mm(torch.diag(s), torch.t(v)))
            this_trace = torch.trace(temp_feature_cov_var)
            if this_trace > 3000.0:
                model.feature_cov_var += 0.0003 * Variable(temp_feature_cov_var / this_trace * 3000.0)# 0.0003
            else:
                model.feature_cov_var += 0.0003 * Variable(temp_feature_cov_var)
            u1, s1, v1 = torch.svd(temp_feature_cov_var1)
            temp_feature_cov_var1 = torch.mm(u1, torch.mm(torch.diag(s1), torch.t(v1)))
            this_trace1 = torch.trace(temp_feature_cov_var1)
            if this_trace1 > 3000.0:
                model.feature_cov_var1 += 0.0003 * Variable(temp_feature_cov_var1 / this_trace1 * 3000.0)  # 0.0003
            else:
                model.feature_cov_var1 += 0.0003 * Variable(temp_feature_cov_var1)
    loss_train_total.append(total_loss.item())
    loss_ts.append(tsloss)

    model.eval()
    model.hidden_cellf = None
    model.hidden_cells = None
    predf, preds=[],[]
    for j, (X_, yf_, ys_) in enumerate(valloader):
        X_ = Variable(X_)
        var_yf_ = Variable(yf_)
        var_ys_ = Variable(ys_)
        valf, vals = model(X_)
        lossf_ = criterion(valf, var_yf_)
        losss_ = criterion(vals, var_ys_)
        valf,vals=valf.data.numpy(),vals.data.numpy()
        valf, vals = np.squeeze(valf), np.squeeze(vals)
        predf.append(valf)
        preds.append(vals)
    predf, preds = np.array(predf), np.array(preds)
    pred_f, pred_s = predf.reshape(288*7, 1), preds.reshape(288*7, 1)
    pred_f = scaler2.inverse_transform(pred_f)
    test_f = scaler2.inverse_transform(val_f)
    pred_s = scaler3.inverse_transform(pred_s)
    test_s = scaler3.inverse_transform(val_s)
    mape_f=0
    mape_s=0
    for k in range(7):
        mape_f+=mape(pred_f[72+288*k:264+288*k],test_f[72+288*k:264+288*k])
        mape_s+=mape(pred_s[72+288*k:264+288*k],test_s[72+288*k:264+288*k])
    mape_f,mape_s=mape_f/7,mape_s/7
    mapef6_22.append(mape_f)
    mapes6_22.append(mape_s)
    val_loss=loss_wf *lossf_+loss_ws*losss_
    loss_val_total.append(val_loss)
    if epo>=80 and epo%10==0:
        torch.save(model.state_dict(),"model_result/"+model_path+str(epo)+".pth")
    # if val_loss<min_valloss:
    #     min_valloss=val_loss
    #     print("min_valloss has been changed!")
    #     torch.save(model.state_dict(),model_path_LowestvalLoss)
    print(f'epoch: {epo:3} total_loss: {total_loss.item():10.6f} tsloss: {tsloss.item():10.6f} '
          f'mtl_loss: {multi_task_loss.item()} lossf: {lossf.item():10.6f} losss: {losss.item():10.6f}'
          f'valloss: {val_loss.item():10.6f} mapef6_22:{mape_f.item():10.6f} mapes6_22:{mape_s.item():10.6f}')

time2=time.time()
print("运行秒数：",time2-time1)

plt.figure()
plt.title('mtl train total loss')
plt.plot(loss_train_total,'b',label='total_loss_train')
plt.legend(loc='upper left')
plt.ylim(0,2*loss_train_total[1])

plt.figure()
plt.title('time series loss')
plt.plot(loss_ts,'b',label='train ts loss')
plt.plot(loss_val_total,'r',label='validation ts loss')
plt.legend(loc='upper left')

plt.figure()
plt.title('val mape')
plt.plot(mapef6_22,'b',label='flow_val')
plt.plot(mapes6_22,'r',label='speed_val')
plt.legend(loc='upper left')
plt.ylim(0,2*mapef6_22[1])

plt.show()





