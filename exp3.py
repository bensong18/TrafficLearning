#由experiment.py修改，同时类似exp1.py,将calss HomoMultiTaskModel放到forward,保留tn prior
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
torch.manual_seed(1)
import tensor_op


# Hyper Parameters
INPUT_SIZE = 7      # input size（输入特征维度）
TIME_STEPS = 8     # time step（步长：使用前TIME_STEP 条数据进行预测）
batch_size = 72

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

#加载数据
data = pd.read_csv("20151101cba-mtl2.csv")
data = data.drop(['Datetime'], axis = 1)
data=np.array(data,dtype='float32')
flow=data[:,0] #drop之后的第0列
speed=data[:,1]
scaler1=MinMaxScaler()
scaler2=MinMaxScaler()
scaler3=MinMaxScaler()
data=scaler1.fit_transform(data)
flow=flow.reshape(-1,1)
speed=speed.reshape(-1,1)
flow=scaler2.fit_transform(flow)
speed=scaler3.fit_transform(speed)
flow_data = data[:,0].reshape(len(data),1)
speed_data = data[:,1].reshape(len(data),1)
##create dataset
data_X,_ = createSamples(data,TIME_STEPS)
_,data_f = createSamples(flow_data,TIME_STEPS)
_,data_s = createSamples(speed_data,TIME_STEPS)

length=396*288-TIME_STEPS
print(length,len(data_X))
t=0
s=batch_size-TIME_STEPS
train_X=data_X[288*(t+350)+s:length-288*(30-t),:,:] # 丢弃前s个数据，因总数据不可被batch整除，而drop_last会丢弃后s个数据，影响更大
train_f=data_f[288*(t+350)+s:length-288*(30-t)]
train_s=data_s[288*(t+350)+s:length-288*(30-t)]
test_X=data_X[length-288*(30-t):length-288*(29-t),:,:]# predict 11.1-11.30
test_f=data_f[length-288*(30-t):length-288*(29-t)]
test_s=data_s[length-288*(30-t):length-288*(29-t)]
print(train_X.shape,train_f.shape,test_X.shape,test_f.shape)

class Time_Series_Data(Dataset):
    def __init__(self, train_x, train_y1,train_y2):
        self.X = train_x
        self.y1 = train_y1
        self.y2 = train_y2
    def __getitem__(self, item):
        x_t = self.X[item]
        y_t1 = self.y1[item]
        y_t2 = self.y2[item]
        return x_t, y_t1,y_t2
    def __len__(self):
        return len(self.X)

traindataset = Time_Series_Data(train_X,train_f,train_s)
trainloader=DataLoader(traindataset,batch_size=batch_size,shuffle=False)#drop_last=True 丢弃最后一个残缺batch
testdataset = Time_Series_Data(test_X,test_f,test_s)
testloader=DataLoader(testdataset,batch_size=batch_size,shuffle=False)

class MRNmodel(nn.Module):
    def __init__(self):
        super(MRNmodel, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=INPUT_SIZE,out_channels=64,kernel_size=1),
            nn.ReLU(),
            nn.Dropout(0.5),
        )
        for m in self.conv1.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=1),
            nn.ReLU(),
            nn.Dropout(0.5),
        )
        # for m in self.conv2.modules():
        #     if isinstance(m, nn.Conv1d):
        #         nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        #         if m.bias is not None:
        #             nn.init.constant_(m.bias, 0)
        self.bilstmf = nn.LSTM(
            input_size=512,
            hidden_size=64,
            num_layers=1,
            batch_first=True,
            bidirectional=True #bilstm
        )
        # self.bilstms = nn.LSTM(
        #     input_size=512,
        #     hidden_size=64,
        #     num_layers=1,
        #     batch_first=True,
        #     bidirectional=True  # bilstm
        # )
        self.dropoutf=nn.Dropout(0.3)
        # self.dropouts=nn.Dropout(0.3)
        self.linearf = nn.Linear(2*64,128) #bilstm应该2*hidden_size
        self.linears = nn.Linear(2*64,128)
        # self.linearf1 = nn.Linear(128, 1)  # bilstm应该2*hidden_size
        # self.linears1 = nn.Linear(128, 1)
        self.linearf.weight.data.normal_(0, 0.01)
        self.linearf.bias.data.fill_(0.0)
        self.linears.weight.data.normal_(0, 0.01)
        self.linears.bias.data.fill_(0.0)
        self.networks = [[nn.Linear(128, 1)] for i in range(2)]
        for i in range(2):
            for layer in self.networks[i]:
                layer.weight.data.normal_(0, 0.01)
                layer.bias.data.fill_(0.0)
        self.networks = [nn.Sequential(*val) for val in self.networks]
        self.hidden_cellf = (torch.zeros(2, batch_size, 64), #(num_layers * num_directions, batch_size, hidden_size)
                             torch.zeros(2, batch_size, 64))
        # self.hidden_cells = (torch.zeros(2, batch_size, 64),  # (num_layers * num_directions, batch_size, hidden_size)
        #                      torch.zeros(2, batch_size, 64))

        self.trade_off = 1
        # self.train_ts_loss = 0
        # self.train_multi_task_loss = 0
        # self.train_total_loss = 0
        # self.print_interval = 1
        self.bottleneck_size = self.networks[0][-1].in_features
        self.iter_num = 0
        self.cov_update_freq = 50

        if self.trade_off > 0:
            # initialize covariance matrix
            self.task_cov = torch.eye(2) #E(2)
            self.class_cov = torch.eye(1) #E(1)
            self.feature_cov = torch.eye(self.bottleneck_size) #生成128*128的单位矩阵E(128)
            self.task_cov_var = Variable(self.task_cov)
            self.class_cov_var = Variable(self.class_cov)
            self.feature_cov_var = Variable(self.feature_cov)

    def forward(self,x):
        convout = self.conv1(x.transpose(1,2))
        convout = self.conv2(convout)#.transpose(1,2)
        lstm_outf, self.hidden_cellf = self.bilstmf(convout.view(len(convout) ,1, -1), self.hidden_cellf) #.view(len(convout) ,1, -1)
        lstm_outs, self.hidden_cellf = self.bilstmf(convout.view(len(convout) ,1, -1), self.hidden_cellf)
        lstm_outf=self.dropoutf(lstm_outf)
        lstm_outs=self.dropoutf(lstm_outs)
        fc_outf = self.linearf(lstm_outf.view(len(convout), -1)) #.view(len(convout), -1)
        fc_outs = self.linears(lstm_outs.view(len(convout), -1))
        predf = self.networks[0](fc_outf)
        preds = self.networks[1](fc_outs)
        # self.iter_num+=1
        return predf,preds

model = MRNmodel()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(),lr=0.01) # learning rate(Adam-0.01,SGD-0.2)
# optimizer = torch.optim.SGD(model.parameters(),lr=0.1, momentum=0.9, weight_decay=0.0005)
print(model)
model.train()

#训练参数
epochs = 15
loss_wf=1.0
loss_ws=1.0
for epo in range(epochs):
    for i, (X, yf,ys) in enumerate(trainloader): # i表示经过了几批batch
        var_x = Variable(X)
        var_yf = Variable(yf)
        var_ys = Variable(ys)
        # forward
        outf,outs = model(var_x)
        # print(outf)
        # print(var_yf)
        lossf = criterion(outf,var_yf)
        losss = criterion(outs,var_ys)
        tsloss = lossf*loss_wf + losss*loss_ws
        if model.trade_off > 0:
            weight_size = model.networks[0][-1].weight.size()
            all_weights = [model.networks[i][-1].weight.view(1, weight_size[0], weight_size[1]) for i in range(2)]
            weights = torch.cat(all_weights, dim=0).contiguous()
            multi_task_loss = tensor_op.MultiTaskLoss(weights, model.task_cov_var, model.class_cov_var,model.feature_cov_var)
            total_loss = tsloss +  model.trade_off* multi_task_loss #各任务的mseloss+trade_off*mtl_loss
            # print(torch.tensor([1.23]).squeeze())
            # print(tsloss) #tensor(0.3352, grad_fn=<AddBackward0>)
            # print(multi_task_loss) #tensor([0.0224], grad_fn=<ViewBackward>)
            # print(total_loss) #tensor([0.3576], grad_fn=<AddBackward0>)...
            # model.train_ts_loss += tsloss.item() #tsloss 的累加和
            # model.train_multi_task_loss += multi_task_loss.item() #multi_task_loss 的累加和
        else:
            total_loss = tsloss
            # model.train_ts_loss += tsloss.item()

        # backward
        optimizer.zero_grad()
        model.hidden_cellf = (torch.zeros(2, batch_size, 64), #(num_layers * num_directions, batch_size, hidden_size)
                              torch.zeros(2, batch_size, 64))
        # model.hidden_cells = (torch.zeros(2, batch_size, 64),  # (num_layers * num_directions, batch_size, hidden_size)
        #                       torch.zeros(2, batch_size, 64))
        model.iter_num+=1
        # v = torch.tensor([1.0],dtype=torch.float)
        # multi_task_loss.backward(v)
        total_loss.backward()
        optimizer.step()

        if model.trade_off > 0 and model.iter_num % model.cov_update_freq == 0:
            # get updated weights #在每个cov_update_freq时更新参数
            weight_size = model.networks[0][-1].weight.size()
            all_weights = [model.networks[i][-1].weight.view(1, weight_size[0], weight_size[1]) for i in
                           range(2)]
            weights = torch.cat(all_weights, dim=0).contiguous()

            ## update cov parameters
            temp_task_cov_var = tensor_op.UpdateCov(weights.data, model.class_cov_var.data, model.feature_cov_var.data)
            temp_class_cov_var = tensor_op.UpdateCov(weights.data.permute(1, 0, 2).contiguous(), model.task_cov_var.data, model.feature_cov_var.data)
            temp_feature_cov_var = tensor_op.UpdateCov(weights.data.permute(2, 0, 1).contiguous(), model.task_cov_var.data,model.class_cov_var.data)

            ## task covariance
            u, s, v = torch.svd(temp_task_cov_var)
            # s = s.cpu().apply_(self.select_func)
            model.task_cov_var = torch.mm(u, torch.mm(torch.diag(s), torch.t(v)))  # s的对角线元素，v的转置
            this_trace = torch.trace(model.task_cov_var)
            if this_trace > 3000.0:  # 3000
                model.task_cov_var = Variable(model.task_cov_var / this_trace * 3000.0)
            else:
                model.task_cov_var = Variable(model.task_cov_var)
            ##class covariance
            u, s, v = torch.svd(temp_class_cov_var)
            # s = s.cpu().apply_(model.select_func).cuda()
            model.class_cov_var = torch.mm(u, torch.mm(torch.diag(s), torch.t(v)))
            this_trace = torch.trace(model.class_cov_var)
            if this_trace > 3000.0:
                model.class_cov_var = Variable(model.class_cov_var / this_trace * 3000.0).cuda()
            else:
                model.class_cov_var = Variable(model.class_cov_var)
            ## feature covariance
            u, s, v = torch.svd(temp_feature_cov_var)
            # s = s.cpu().apply_(self.select_func).cuda()
            temp_feature_cov_var = torch.mm(u, torch.mm(torch.diag(s), torch.t(v)))
            this_trace = torch.trace(temp_feature_cov_var)
            if this_trace > 3000.0:
                model.feature_cov_var += 0.0003 * Variable(temp_feature_cov_var / this_trace * 3000.0)  # 0.0003
            else:
                model.feature_cov_var += 0.0003 * Variable(temp_feature_cov_var)  # .cuda()
    print(f'epoch: {epo+1:3} total_loss: {total_loss.item():10.6f} tsloss: {tsloss.item():10.6f} multi_task_loss: {multi_task_loss.item()}')


    # print(model.iter_num)
    # if model.iter_num % model.print_interval == 0:
    #     model.train_total_loss = model.train_ts_loss + model.train_multi_task_loss  # 两者累加和之和
    #     print(
    #         "Iter {:05d}, Average TimeSeries Loss: {:.4f}; Average MultiTask Loss: {:.4f}; Average Training Loss: {:.4f}".format(
    #             model.iter_num, model.train_ts_loss / float(model.print_interval),
    #                             model.train_multi_task_loss / float(model.print_interval),
    #                             model.train_total_loss / float(model.print_interval)))
    #     model.train_ts_loss = 0
    #     model.train_multi_task_loss = 0
    #     model.train_total_loss = 0

model.eval()

##predict,预测
pred_f,pred_s=[],[]
for j, (X_, yf_,ys_) in enumerate(testloader):
    X_ = Variable(X_)
    predf,preds = model(X_)
    predf = predf.data.numpy()
    predf = np.squeeze(predf)
    pred_f.append(predf)
    preds = preds.data.numpy()
    preds = np.squeeze(preds)
    pred_s.append(preds)
pred_f=np.array(pred_f)
# print(pred_f.shape)
pred_f=pred_f.reshape(288,1)
pred_s=np.array(pred_s)
pred_s=pred_s.reshape(288,1)

pred_f=scaler2.inverse_transform(pred_f)
test_f=scaler2.inverse_transform(test_f)
pred_s=scaler3.inverse_transform(pred_s)
test_s=scaler3.inverse_transform(test_s)

print('pred_f[:10]',pred_f[:10],'pred_f[-10:]',pred_f[-10:])
print('test_f[:10]',test_f[:10],'test_f[-10:]',test_f[-10:])

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

fig1=plt.figure()
ax1 = fig1.add_subplot(111)
plt.title('traffic flow prediction')
plt.plot(test_f,'b',label='real flow')
plt.plot(pred_f,'r', label='predict flow')
plt.legend(loc='upper left')
xticks = [0,48,96,144,192,240,288]
xlabels = ['0:00','4:00','8:00','12:00','16:00','20:00','24:00']
ax1.set_xticks(xticks)
ax1.set_xticklabels(xlabels)#, rotation='45') #rotation='vertical'
plt.xlim(0,288)
plt.ylim(0,530)

fig2=plt.figure()
ax2 = fig2.add_subplot(111)
plt.title('traffic speed prediction')
plt.plot(test_s,'b',label='real speed')
plt.plot(pred_s,'r', label='predict speed')
plt.legend(loc='upper left')
ax2.set_xticks(xticks)
ax2.set_xticklabels(xlabels)
plt.xlim(0,288)
plt.ylim(min(test_s)-45,max(test_s+45))
plt.show()