#coding=gbk
import pandas as pd
import numpy as np
import matplotlib as plt
from collections import defaultdict,Counter
import sys
sys.setrecursionlimit(1000000)
import scipy.stats as stats
data=pd.read_csv(r'BDAM\csgo_round_snapshots.csv')#�ܴ����ʱ������
dataf=pd.DataFrame(data)
dataf=dataf.loc[:,['ct_money','t_money','round_winner']]#��ȡ�����Ϣ
dataf2=pd.DataFrame(data)
dataf2=dataf2.loc[:,['ct_money','t_money','round_winner']]

#˫�������滻
for i in range(len(dataf)):
    if dataf.loc[i,'round_winner']=='CT':
        dataf.loc[i,'round_winner']=1
    else:
        dataf.loc[i,'round_winner']=2
re=dataf.describe()
print('The rough statistic:\n',re,end='\n')

def classing(side):
    dist=2000
    for i in range(len(dataf)):
        if 0<=dataf.loc[i,side]<dist:
            dataf.loc[i,side]='H'
        elif dist<=dataf.loc[i,side]<(2*dist):
            dataf.loc[i,side]='G'
        elif (2*dist)<=dataf.loc[i,side]<(3*dist):
            dataf.loc[i,side]='F'
        elif (3*dist)<=dataf.loc[i,side]<(4*dist):
            dataf.loc[i,side]='E'
        elif (4*dist)<=dataf.loc[i,side]<(5*dist):
            dataf.loc[i,side]='D'
        elif (5*dist)<=dataf.loc[i,side]<(6*dist):
            dataf.loc[i,side]='C'
        elif (6*dist)<=dataf.loc[i,side]<(7*dist):
            dataf.loc[i,side]='B'
        elif (7*dist)<=dataf.loc[i,side]:
            dataf.loc[i,side]='A'
    return dataf

#���ݷ���
classing('ct_money')
classing('t_money')
CTcleaned=pd.DataFrame([[i for i in dataf['ct_money']],[i for i in dataf['round_winner']]],index=['CT_money','round_winner']).T
Tcleaned=pd.DataFrame([[i for i in dataf['t_money']],[i for i in dataf['round_winner']]],index=['T_money','round_winner']).T

#��dataframeת����array
list1=[]
list2=[]
for i in range(len(CTcleaned)):
    list1.append(CTcleaned.loc[i,'CT_money'])   
for i in range(len(CTcleaned)):
    list2.append(CTcleaned.loc[i,'round_winner'])
listCT=list(zip(list2,list1))
list1=[]
list2=[]
for i in range(len(Tcleaned)):
    list1.append(Tcleaned.loc[i,'T_money'])   
for i in range(len(Tcleaned)):
    list2.append(Tcleaned.loc[i,'round_winner'])
listT=list(zip(list2,list1))

#���㱴Ҷ˹����
class NBayes:
    def __init__(self, lambda_=1):
        self.lambda_ = lambda_  # ��Ҷ˹���Ʒ�������lambda
        self.p_prior = {}  # ģ�͵��������, ע�������������ʲ���ָԤ����Ϊ�趨��������ʣ�������Ҫ���Ƶ�P(y=Ck)
        self.p_condition = {}  # ģ�͵���������

    def fit(self, X_data, y_data):
        N = y_data.shape[0]
        # ������������P(y=Ck)�ĺ�����ʣ��趨�������Ϊ���ȷֲ�
        c_y = Counter(y_data)
        K = len(c_y)
        for key, val in c_y.items():
            self.p_prior[key] = (val + self.lambda_) / (N + K * self.lambda_)
        # ������������P(Xd=a|y=Ck)�ĺ�����ʣ�ͬ���������Ϊ���ȷֲ�
        for d in range(X_data.shape[1]):  # �Ը���ά�ȷֱ���д���
            Xd_y = defaultdict(int)
            vector = X_data[:, d]
            Sd = len(np.unique(vector))
            for xd, y in zip(vector, y_data): # ����Xd�����ǳ��������ݼ�D�е�������ʼ�ʹ�ü�����Ȼ����Ҷû�и���Ϊ0�����
                Xd_y[(xd, y)] += 1
            for key, val in Xd_y.items():
                self.p_condition[(d, key[0], key[1])] = (val + self.lambda_) / (c_y[key[1]] + Sd * self.lambda_)
        return

    def predict(self, X):
        p_post = defaultdict()
        for y, py in self.p_prior.items():
            p_joint = py  # ���ϸ��ʷֲ�
            for d, Xd in enumerate(X):
                p_joint *= self.p_condition[(d, Xd, y)]  # ���������Լ���
            p_post[y] = p_joint  # ��ĸP(X)��ͬ����ֱ�Ӵ洢���ϸ��ʷֲ�����
        return max(p_post, key=p_post.get)


if __name__ == '__main__':
    data = np.array(listCT)#������ͬ��Ҷ˹��������������ࣺlistCT��listT
    X_data = data[:, :-1]
    y_data = data[:, -1]
    clf = NBayes(lambda_=1)
    clf.fit(X_data, y_data)
    print(clf.p_prior, '\n', clf.p_condition,end='\n')
    pd.DataFrame(clf.p_prior,index=[range(8)]).to_csv("bayes.csv")
#����list�ĺ�׺���˴����ر�Ҷ˹������������CT������²�ͬ����״���ķֲ������ڲ�ͬ����������CT��ʤ�ʣ���CTΪ������������ֻ�������ڵڶ�λ����Ϊ1���������CTΪ����T��Ϊ2��

#���������Լ���
#ͳ�Ƹ����������¸���ʤ����
def cal(side):
    dataset=dataf.groupby([side])
    a=[]
    for name ,group in dataset:
        num=group['round_winner'].value_counts()
        if side.find('CT')>=0:
            b=num[1]
        else:
            b=num[2]
        list1=[name,b]
        a.append(list1)
    a=pd.DataFrame(a)
    return a
#�ϲ����
CTindependenceExamData=cal('ct_money')
CTindependenceExamData.index=CTindependenceExamData.iloc[:,0]
CTindependenceExamData.drop([0],axis=1,inplace=True)
CTindependenceExamData.columns=['CT']
CTindependenceExamData=pd.Series([i for i in CTindependenceExamData['CT']],index=['A','B','C','D','E','F','G','H'])
TindependenceExamData=cal('t_money')
TindependenceExamData.index=TindependenceExamData.iloc[:,0]
TindependenceExamData.drop([0],axis=1,inplace=True)
TindependenceExamData.columns=["T"]
TindependenceExamData=pd.Series([i for i in TindependenceExamData['T']],index=['A','B','C','D','E','F','G','H'])
FinalIndependenceExamData=pd.concat([CTindependenceExamData,TindependenceExamData],axis=1)
FinalIndependenceExamData.columns=['CT','T']
FinalIndependenceExamData['Sum']=FinalIndependenceExamData['CT']+FinalIndependenceExamData['T']
#��������
def chi2_test(df):
    s,r=len(df.columns),len(df.index)
    x=[]
    for i in range(r):
        for j in range(s):
            nij=df.loc[df.index[i],df.columns[j]]
            mij=sum(df.loc[df.index[i]])*sum(df[df.columns[j]])/df.sum().sum()
            x.append((nij-mij)**2/mij)
    p=float(stats.chi2.sf(sum(x),(r-1)*(s-1)))
    return {'ͳ��ֵ':sum(x),'���ɶ�':(r-1)*(s-1),'pֵ':p}
ec=chi2_test(FinalIndependenceExamData)
print(ec)
pd.DataFrame(ec,index=[0]).to_csv('Economy.csv')


#����Person���ϵ��
#�����滻
def classing1(side):
    dist=2000
    for i in range(len(dataf2)):
        if 0<=dataf2.loc[i,side]<dist:
            dataf2.loc[i,side]=1
        elif dist<=dataf2.loc[i,side]<(2*dist):
            dataf2.loc[i,side]=2
        elif (2*dist)<=dataf2.loc[i,side]<(3*dist):
            dataf2.loc[i,side]=3
        elif (3*dist)<=dataf2.loc[i,side]<(4*dist):
            dataf2.loc[i,side]=4
        elif (4*dist)<=dataf2.loc[i,side]<(5*dist):
            dataf2.loc[i,side]=5
        elif (5*dist)<=dataf2.loc[i,side]<(6*dist):
            dataf2.loc[i,side]=6
        elif (6*dist)<=dataf2.loc[i,side]<(7*dist):
            dataf2.loc[i,side]=7
        elif (7*dist)<=dataf2.loc[i,side]:
            dataf2.loc[i,side]=8
    return dataf2

#���ݷ���
classing1('ct_money')
classing1('t_money')
CTcleaned=pd.DataFrame([[i for i in dataf['ct_money']],[i for i in dataf['round_winner']]],index=['CT_money','round_winner']).T
Tcleaned=pd.DataFrame([[i for i in dataf['t_money']],[i for i in dataf['round_winner']]],index=['T_money','round_winner']).T

#ͳ�Ƹ����������¸���ʤ����
def cal(side):
    dataset=dataf2.groupby([side])
    a=[]
    for name ,group in dataset:
        num=group['round_winner'].value_counts()
        if side.find('ct')>=0:
            b=num['CT']
        else:
            b=num['T']
        list1=[name,b]
        a.append(list1)
    a=pd.DataFrame(a)
    return a
#�ϲ����
CTindependenceExamData=cal('ct_money')
CTindependenceExamData.index=CTindependenceExamData.iloc[:,0]
CTindependenceExamData.columns=['CT_money','win_round']
TindependenceExamData=cal('t_money')
TindependenceExamData.index=TindependenceExamData.iloc[:,0]
TindependenceExamData.columns=["T_money",'win_round']

#Ƥ��ɭϵ���������
def pearsonRelation(data,side):
    u1,u2 = data[side].mean(),data['win_round'].mean()  # �����ֵ
    std1,std2 = data[side].std(),data['win_round'].std()  # �����׼��
    #�������ϵ����������
    data['(x-u1)*(y-u2)'] = (data[side] - u1) * (data['win_round'] - u2)
    data['(x-u1)**2'] = (data[side] - u1)**2
    data['(y-u2)**2'] = (data['win_round'] - u2)**2
    print(data)
    print('------')
    #�������ϵ��
    r = data['(x-u1)*(y-u2)'].sum() / (np.sqrt(data['(x-u1)**2'].sum() * data['(y-u2)**2'].sum()))
    print('Pearson���ϵ��Ϊ��%.4f' % r)
print('T person relation:')
pearsonRelation(TindependenceExamData,'T_money')
print('CT person relation:')
pearsonRelation(CTindependenceExamData, 'CT_money')
#Ƥ��ɭ���ϵ��˵����Ƥ��ɭ���ϵ������˵���������Ƿ�������Թ�ϵ
#���ϣ�������������Ӱ���ϵ�����������Լ���