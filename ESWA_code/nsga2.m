%% 神经网络+NSGAII优化
clear;
clc;
num=xlsread('1_201511-201611各分析.xlsx');   %加载数据
num(1,:) = [];
num(:,1) = [];
num(:,6:end) = [];
num(369:end,:) = [];
num = num';
%训练集
input_train=num(1:4,1:round(size(num,1)*0.9));
output_train=num(5,1:round(size(num,1)*0.9));
input_test=num(1:4,round(size(num,1)*0.9)+1:end);
net=newff(input_train,output_train,5);          %BP神经网络
net.trainParam.epochs=200;
net.trainParam.lr=0.1;
net.trainParam.goal=0.000001;
net=train(net,input_train,output_train);
output_test=sim(net,input_test);
output_testf=num(5,round(size(num,1)*0.9)+1:end);
error=zeros(1,5);
for i=1:length(output_test)
    error(i)=abs(output_test(i)-output_testf(i))/output_testf(i);
end
error=mean(error);
%% 优化确定权重顺序
%定义GA参数
NIND=40;               %种群大小
MAXGEN=100;            %最大遗传代数
GGAP=0.95;             %代沟
px=0.7;                %交叉概率
pm=0.01;               %变异概率
trace=zeros(21,MAXGEN);%寻优结果初始值
[Chrom,Lind,BaseV]=crtbp(NIND,[2*ones(1,4),5,2,2,4]);
%优化
gen=0;                 %代数计数器
XY=Chrom;%初始种群的十进制转换
ObjV=sim(net,XY(:,1:4)'.*XY(:,5:8)')';    %计算目标函数值
while gen<MAXGEN
    FitnV=ranking(ObjV); %分配适应度值
    SelCh=select('sus',Chrom,FitnV,GGAP);   %选择
    SelCh=recombin('xovsp',SelCh,px);       %重组
    SelCh=mut(SelCh,pm,[2*ones(1,4),5,2,2,4]);  %变异
    XY=SelCh;                 %子代个体的十进制转换
    ObjVSel=sim(net,XY(:,1:4)'.*XY(:,5:8)')';  %计算子代的目标函数值
    [Chrom,ObjV]=reins(Chrom,SelCh,1,1,ObjV,ObjVSel);  %重插入子代到父代，得到新种群
    XY=Chrom;
    gen=gen+1;
    %获取每代的最优解及其序号
    [Y,I]=min(ObjV);
    trace(1:8,gen)=XY(I,:);                %记下每代最优值
    trace(9,gen)=Y;                        %记下每代最优值
end
%% 画进化图
figure;
plot(1:MAXGEN,trace(9,:));
grid on
xlabel('遗传代数')
ylabel('解的变化')
title('进化过程')
disp('最优变量(相关性)：');
trace(1:4,end)'
%% 绘制Pareto前沿
[x,y] = multiobj(m,n);
% x = 1:4;
% y = [0.3636,0.2671,0.2154,0.1897];
figure;
plot(x,y,'ko','Markersize',20,'MarkerFacecolor','b','MarkerEdgeColor','b');
grid on;
xlabel('Number of factors');
ylabel('Error');
set(gca,'FontSize',30,'FontName','Times New Roman');
axis([0 5 0 0.4]);