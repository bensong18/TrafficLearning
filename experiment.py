# -*- coding: utf-8 -*-
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
LR = 0.001           # learning rate
TIME_STEPS = 8     # time step（步长：使用前TIME_STEP 条数据进行预测）
batch_size = 72

class Sharelayer(nn.Module):
    def __init__(self):
        super(Sharelayer, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=INPUT_SIZE, out_channels=64, kernel_size=1),
            # nn.BatchNorm1d(200),
            nn.ReLU(),
            nn.Dropout(0.5),
            # nn.MaxPool1d(kernel_size=3)
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=1),
            # nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Dropout(0.5),
            # nn.MaxPool1d(kernel_size=3)
        )

        self.bilstmf = nn.LSTM(
            input_size=512,
            hidden_size=64,
            num_layers=1,
            batch_first=True,
            bidirectional=True  # bilstm
        )
        # self.bilstms = nn.LSTM(
        #     input_size=512,
        #     hidden_size=64,
        #     num_layers=1,
        #     batch_first=True,
        #     bidirectional=True  # bilstm
        # )
        self.dropoutf = nn.Dropout(0.3)
        # self.dropouts = nn.Dropout(0.3)
        # self.linearf = nn.Linear(2 * 64, 1)  # bilstm应该2*hidden_size
        # self.linears = nn.Linear(2 * 64, 1)
        # self.linearf1 = nn.Linear(1, 1)  # bilstm应该2*hidden_size
        # self.linears1 = nn.Linear(1, 1)
        self.hidden_cellf = (torch.zeros(2, batch_size, 64),  # (num_layers * num_directions, batch_size, hidden_size)
                             torch.zeros(2, batch_size, 64))
        # self.hidden_cells = (torch.zeros(2, batch_size, 64),  # (num_layers * num_directions, batch_size, hidden_size)
        #                      torch.zeros(2, batch_size, 64))

    def forward(self, x):
        convout = self.conv1(x.transpose(1, 2))
        convout = self.conv2(convout)  # .transpose(1,2)
        print(len(convout))
        lstm_outf, self.hidden_cellf = self.bilstmf(convout.view(len(convout), 1, -1),
                                                    self.hidden_cellf)  # .view(len(convout) ,1, -1)
        # lstm_outs, self.hidden_cells = self.bilstms(convout.view(len(convout), 1, -1), self.hidden_cells)
        lstm_outf = self.dropoutf(lstm_outf)
        lstm_outf = lstm_outf.view(len(convout), -1)
        # lstm_outs = self.dropouts(lstm_outs)
        # predf = self.linearf(lstm_outf.view(len(convout), -1))  # .view(len(convout), -1)
        # preds = self.linears(lstm_outs.view(len(convout), -1))
        # predf = self.linearf1(predf)
        # preds = self.linearf1(preds)
        # return predf, preds
        return lstm_outf

network_dict = {"Sharelayer": Sharelayer}


class HomoMultiTaskModel(object):
    # num_tasks: number of tasks
    # network_name: the base model used, add new network name in the above 'network_dict'
    # output_num: the output dimension of all the tasks
    # gpus: gpu id used (list)
    # file_out: log file
    # trade_off: the trade_off between multitask loss and task loss
    # optim_param: optimizer parameters
    def __init__(self, num_tasks, network_name, output_num, trade_off=1.0,
                 optim_param={"init_lr": 0.00003, "gamma": 0.3, "power": 0.75, "stepsize": 5}):
        def select_func(x):
            if x > 0.1:
                return 1. / x
            else:
                return x

        # self.file_out = file_out
        # threshold function in filtering too small singular value
        self.select_func = select_func

        self.trade_off = trade_off
        self.train_cross_loss = 0
        self.train_multi_task_loss = 0
        self.train_total_loss = 0
        self.print_interval = 5

        # covariance update frequency (one every #param iter)
        self.cov_update_freq = 10

        # construct multitask model with shared part and related part
        self.num_tasks = num_tasks
        self.network_name = network_name
        self.output_num = output_num # 1
        # self.num_gpus = len(gpus)
        # print(self.output_num)
        self.shared_layers = network_dict[self.network_name]()  # layers shared
        self.networks = [[nn.Linear(128, self.output_num)] for i in
                         range(self.num_tasks)]  #layers not shared but related,生成任务数分支的fc
        for i in range(self.num_tasks):
            for layer in self.networks[i]:
                layer.weight.data.normal_(0, 0.01)
                layer.bias.data.fill_(0.0)
        self.networks = [nn.Sequential(*val) for val in self.networks]  #  ???

        self.bottleneck_size = self.networks[0][-1].in_features #128

        # self.shared_layers = nn.DataParallel(self.shared_layers)
        # self.networks = [nn.DataParallel(network) for network in self.networks]

        # construct optimizer
        parameter_dict = [{"params": self.shared_layers.parameters(), "lr": 0}]#self.shared_layers.module.parameters()
        parameter_dict += [{"params": self.networks[i].parameters(), "lr": 1} for i in range(self.num_tasks)]#10
        self.optimizer = torch.optim.SGD(parameter_dict, lr=1, momentum=0.9, weight_decay=0.0005)  # parameter_dict??
        self.parameter_lr_dict = []
        for param_group in self.optimizer.param_groups:
            self.parameter_lr_dict.append(param_group["lr"])
        self.optim_param = {"init_lr": 0.00003, "gamma": 0.3, "power": 0.75, "stepsize": 5}
        for val in optim_param:
            self.optim_param[val] = optim_param[val]

        if self.trade_off > 0:
            # initialize covariance matrix
            self.task_cov = torch.eye(num_tasks) #E(2)
            self.class_cov = torch.eye(output_num) #E(1)
            self.feature_cov = torch.eye(self.bottleneck_size) #生成128*128的单位矩阵E(128)

            self.task_cov_var = Variable(self.task_cov)
            self.class_cov_var = Variable(self.class_cov)
            self.feature_cov_var = Variable(self.feature_cov)

        self.criterion = nn.MSELoss()  # mse
        self.iter_num = 1

    def optimize_model(self, input_list, label_list):
        # update learning rate
        current_lr = self.optim_param["init_lr"] * (
                    self.optim_param["gamma"] ** (self.iter_num // self.optim_param["stepsize"]))
        i = 0
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = current_lr * self.parameter_lr_dict[i]
            i += 1

        # timeseries loss
        for i in range(self.num_tasks):
            self.networks[i].train(True)
        self.shared_layers.train(True)

        self.optimizer.zero_grad()
        self.shared_layers.hidden_cellf = (torch.zeros(2, batch_size, 64), #(num_layers * num_directions, batch_size, hidden_size)
                                           torch.zeros(2, batch_size, 64))
        # concat_input = torch.cat(input_list, dim=0)
        feature_out = self.shared_layers(input_list)
        output_list = [self.networks[i](feature_out.narrow(0, i * 36, 36)) for i in #batch_size/2 ??
                       range(self.num_tasks)]
        losses = [self.criterion(output_list[i], label_list[i]) for i in range(self.num_tasks)]
        classifier_loss = sum(losses) #time series loss

        # multitask loss
        if self.trade_off > 0:
            # weight_size = self.networks[0].size()
            # weight_size = self.networks[0].module[-1].weight.size()
            all_weights = [feature_out.view(1, 72, 128) for i in range(self.num_tasks)] # ???weight size 如何确定?
            weights = torch.cat(all_weights, dim=0).contiguous()

            multi_task_loss = tensor_op.MultiTaskLoss(weights, self.task_cov_var, self.class_cov_var,self.feature_cov_var)
            total_loss = classifier_loss + self.trade_off * multi_task_loss
            self.train_cross_loss += classifier_loss.data[0]
            self.train_multi_task_loss += multi_task_loss.data[0]
        else:
            total_loss = classifier_loss
            self.train_cross_loss += classifier_loss.data[0]
        # update network parameters
        total_loss.backward()
        self.optimizer.step()

        if self.trade_off > 0 and self.iter_num % self.cov_update_freq == 0:
            # get updated weights
            weight_size = self.networks[0].module[-1].weight.size()
            all_weights = [self.networks[i].module[-1].weight.view(1, weight_size[0], weight_size[1]) for i in
                           range(self.num_tasks)]
            weights = torch.cat(all_weights, dim=0).contiguous()

            # update cov parameters
            temp_task_cov_var = tensor_op.UpdateCov(weights.data, self.class_cov_var.data, self.feature_cov_var.data)

            # temp_class_cov_var = tensor_op.UpdateCov(weights.data.permute(1, 0, 2).contiguous(), self.task_cov_var.data, self.feature_cov_var.data)
            # temp_feature_cov_var = tensor_op.UpdateCov(weights.data.permute(2, 0, 1).contiguous(), self.task_cov_var.data, self.class_cov_var.data)

            # task covariance
            u, s, v = torch.svd(temp_task_cov_var)
            s = s.cpu().apply_(self.select_func)
            self.task_cov_var = torch.mm(u, torch.mm(torch.diag(s), torch.t(v)))
            this_trace = torch.trace(self.task_cov_var)
            if this_trace > 3000.0:
                self.task_cov_var = Variable(self.task_cov_var / this_trace * 3000.0)
            else:
                self.task_cov_var = Variable(self.task_cov_var)
            # uncomment to use the other two covariance
            '''
            # class covariance
            u, s, v = torch.svd(temp_class_cov_var)
            s = s.cpu().apply_(self.select_func).cuda()
            self.class_cov_var = torch.mm(u, torch.mm(torch.diag(s), torch.t(v)))
            this_trace = torch.trace(self.class_cov_var)
            if this_trace > 3000.0:        
                self.class_cov_var = Variable(self.class_cov_var / this_trace * 3000.0).cuda()
            else:
                self.class_cov_var = Variable(self.class_cov_var).cuda()
            # feature covariance
            u, s, v = torch.svd(temp_feature_cov_var)
            s = s.cpu().apply_(self.select_func).cuda()
            temp_feature_cov_var = torch.mm(u, torch.mm(torch.diag(s), torch.t(v)))
            this_trace = torch.trace(temp_feature_cov_var)
            if this_trace > 3000.0:        
                self.feature_cov_var += 0.0003 * Variable(temp_feature_cov_var / this_trace * 3000.0).cuda()
            else:
                self.feature_cov_var += 0.0003 * Variable(temp_feature_cov_var).cuda()
            '''
        # self.iter_num += 1
        # if self.iter_num % self.print_interval == 0:
        #     self.train_total_loss = self.train_cross_loss + self.train_multi_task_loss
        #     print(
        #         "Iter {:05d}, Average Cross Entropy Loss: {:.4f}; Average MultiTask Loss: {:.4f}; Average Training Loss: {:.4f}".format(
        #             self.iter_num, self.train_cross_loss / float(self.print_interval),
        #             self.train_multi_task_loss / float(self.print_interval),
        #             self.train_total_loss / float(self.print_interval)))
        #     self.file_out.write(
        #         "Iter {:05d}, Average Cross Entropy Loss: {:.4f}; Average MultiTask Loss: {:.4f}; Average Training Loss: {:.4f}\n".format(
        #             self.iter_num, self.train_cross_loss / float(self.print_interval),
        #             self.train_multi_task_loss / float(self.print_interval),
        #             self.train_total_loss / float(self.print_interval)))
        #     self.file_out.flush()
        #     self.train_cross_loss = 0
        #     self.train_multi_task_loss = 0
        #     self.train_total_loss = 0

    def run_model(self, input_, i):
        self.shared_layers.train(False)
        self.networks[i].train(False)
        output = self.networks[i](self.shared_layers(input_))
        return output

# import configparser
# config = configparser.ConfigParser()
# config["gpus"] = range(len(os.environ["CUDA_VISIBLE_DEVICES"].split(",")))
model=HomoMultiTaskModel(num_tasks=2, network_name="Sharelayer",output_num=1,trade_off=1,optim_param={"init_lr":0.00003, "gamma":0.3, "power":0.75, "stepsize":5})
print(model)
# model.eval()

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
data = data.drop(['Datetime'], axis = 1)
data=np.array(data,dtype='float32')
# flow=data[:,0] #drop之后的第0列
# speed=data[:,1]
label = data[:,[0,1]]
#print(data)
scaler1=MinMaxScaler()
scaler2=MinMaxScaler()
scaler3=MinMaxScaler()
data=scaler1.fit_transform(data)
# flow=flow.reshape(-1,1)
# speed=speed.reshape(-1,1)
# flow=scaler2.fit_transform(flow)
# speed=scaler3.fit_transform(speed)
label=scaler2.fit_transform(label)
# flow_data = data[:,0].reshape(len(data),1)
# speed_data = data[:,1].reshape(len(data),1)
###create dataset
data_X,_ = createSamples(data,TIME_STEPS)
# _,data_f = createSamples(flow_data,TIME_STEPS)
# _,data_s = createSamples(speed_data,TIME_STEPS)
_,data_y = createSamples(label,TIME_STEPS)

length=396*288-TIME_STEPS
print(length,len(data_X))
t=0
s=batch_size-TIME_STEPS
# s=0
train_X=data_X[288*(t+350)+s:length-288*(30-t),:,:] # 丢弃前s个数据，因总数据不可被batch整除，而drop_last会丢弃后s个数据，影响更大
# train_f=data_f[288*(t+350)+s:length-288*(30-t)]
# train_s=data_s[288*(t+350)+s:length-288*(30-t)]
train_y=data_y[288*(t+350)+s:length-288*(30-t)]
test_X=data_X[length-288*(30-t):length-288*(29-t),:,:]# predict 11.1-11.30
# test_f=data_f[length-288*(30-t):length-288*(29-t)]
# test_s=data_s[length-288*(30-t):length-288*(29-t)]
test_y=data_y[length-288*(30-t):length-288*(29-t)]
print(train_X.shape,train_y.shape,test_X.shape,test_y.shape)

class Time_Series_Data(Dataset):
    def __init__(self, train_x, train_y1):
        self.X = train_x
        self.y1 = train_y1
        # self.y2 = train_y2
    def __getitem__(self, item):
        x_t = self.X[item]
        y_t1 = self.y1[item]
        # y_t2 = self.y2[item]
        return x_t, y_t1#,y_t2
    def __len__(self):
        return len(self.X)

traindataset = Time_Series_Data(train_X,train_y)#train_f,train_s
trainloader=DataLoader(traindataset,batch_size=batch_size,shuffle=False)#drop_last=True 丢弃最后一个残缺batch
testdataset = Time_Series_Data(test_X,test_y)
testloader=DataLoader(testdataset,batch_size=batch_size,shuffle=False)

#训练参数
epochs = 10
for epo in range(epochs):
    for i, (X, y) in enumerate(trainloader): # i表示经过了几批batch
        var_x = Variable(X)
        var_y = Variable(y)
        model.optimize_model(var_x,var_y)

##predict,预测
pred_y=[]
for i in range(2):
    for j, (X_, y_) in enumerate(testloader):
        X_ = Variable(X_)
        predy = model.run_model(X_,i)
        predy = predy.data.numpy()
        predy = np.squeeze(predy)
        #predf = np.squeeze(predf)
        pred_y.append(predy)

print(pred_y)
pred_y=np.array(pred_y)
print(pred_y.shape)
# pred_f=pred_f.reshape(288,1)
# pred_s=np.array(pred_s)
# pred_s=pred_s.reshape(288,1)



