%% ������+NSGAII�Ż�
clear;
clc;
num=xlsread('1_201511-201611������.xlsx');   %��������
num(1,:) = [];
num(:,1) = [];
num(:,6:end) = [];
num(369:end,:) = [];
num = num';
%ѵ����
input_train=num(1:4,1:round(size(num,1)*0.9));
output_train=num(5,1:round(size(num,1)*0.9));
input_test=num(1:4,round(size(num,1)*0.9)+1:end);
net=newff(input_train,output_train,5);          %BP������
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
%% �Ż�ȷ��Ȩ��˳��
%����GA����
NIND=40;               %��Ⱥ��С
MAXGEN=100;            %����Ŵ�����
GGAP=0.95;             %����
px=0.7;                %�������
pm=0.01;               %�������
trace=zeros(21,MAXGEN);%Ѱ�Ž����ʼֵ
[Chrom,Lind,BaseV]=crtbp(NIND,[2*ones(1,4),5,2,2,4]);
%�Ż�
gen=0;                 %����������
XY=Chrom;%��ʼ��Ⱥ��ʮ����ת��
ObjV=sim(net,XY(:,1:4)'.*XY(:,5:8)')';    %����Ŀ�꺯��ֵ
while gen<MAXGEN
    FitnV=ranking(ObjV); %������Ӧ��ֵ
    SelCh=select('sus',Chrom,FitnV,GGAP);   %ѡ��
    SelCh=recombin('xovsp',SelCh,px);       %����
    SelCh=mut(SelCh,pm,[2*ones(1,4),5,2,2,4]);  %����
    XY=SelCh;                 %�Ӵ������ʮ����ת��
    ObjVSel=sim(net,XY(:,1:4)'.*XY(:,5:8)')';  %�����Ӵ���Ŀ�꺯��ֵ
    [Chrom,ObjV]=reins(Chrom,SelCh,1,1,ObjV,ObjVSel);  %�ز����Ӵ����������õ�����Ⱥ
    XY=Chrom;
    gen=gen+1;
    %��ȡÿ�������Ž⼰�����
    [Y,I]=min(ObjV);
    trace(1:8,gen)=XY(I,:);                %����ÿ������ֵ
    trace(9,gen)=Y;                        %����ÿ������ֵ
end
%% ������ͼ
figure;
plot(1:MAXGEN,trace(9,:));
grid on
xlabel('�Ŵ�����')
ylabel('��ı仯')
title('��������')
disp('���ű���(�����)��');
trace(1:4,end)'
%% ����Paretoǰ��
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