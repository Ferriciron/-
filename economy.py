#coding=gbk
import pandas as pd
import numpy as np
import matplotlib as plt
from collections import defaultdict,Counter
import sys
sys.setrecursionlimit(1000000)
import scipy.stats as stats
data=pd.read_csv(r'BDAM\csgo_round_snapshots.csv')#跑代码的时候改这个
dataf=pd.DataFrame(data)
dataf=dataf.loc[:,['ct_money','t_money','round_winner']]#提取相关信息
dataf2=pd.DataFrame(data)
dataf2=dataf2.loc[:,['ct_money','t_money','round_winner']]

#双方数据替换
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

#数据分离
classing('ct_money')
classing('t_money')
CTcleaned=pd.DataFrame([[i for i in dataf['ct_money']],[i for i in dataf['round_winner']]],index=['CT_money','round_winner']).T
Tcleaned=pd.DataFrame([[i for i in dataf['t_money']],[i for i in dataf['round_winner']]],index=['T_money','round_winner']).T

#将dataframe转换成array
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

#计算贝叶斯概率
class NBayes:
    def __init__(self, lambda_=1):
        self.lambda_ = lambda_  # 贝叶斯估计方法参数lambda
        self.p_prior = {}  # 模型的先验概率, 注意这里的先验概率不是指预先人为设定的先验概率，而是需要估计的P(y=Ck)
        self.p_condition = {}  # 模型的条件概率

    def fit(self, X_data, y_data):
        N = y_data.shape[0]
        # 后验期望估计P(y=Ck)的后验概率，设定先验概率为均匀分布
        c_y = Counter(y_data)
        K = len(c_y)
        for key, val in c_y.items():
            self.p_prior[key] = (val + self.lambda_) / (N + K * self.lambda_)
        # 后验期望估计P(Xd=a|y=Ck)的后验概率，同样先验概率为均匀分布
        for d in range(X_data.shape[1]):  # 对各个维度分别进行处理
            Xd_y = defaultdict(int)
            vector = X_data[:, d]
            Sd = len(np.unique(vector))
            for xd, y in zip(vector, y_data): # 这里Xd仅考虑出现在数据集D中的情况，故即使用极大似然估计叶没有概率为0的情况
                Xd_y[(xd, y)] += 1
            for key, val in Xd_y.items():
                self.p_condition[(d, key[0], key[1])] = (val + self.lambda_) / (c_y[key[1]] + Sd * self.lambda_)
        return

    def predict(self, X):
        p_post = defaultdict()
        for y, py in self.p_prior.items():
            p_joint = py  # 联合概率分布
            for d, Xd in enumerate(X):
                p_joint *= self.p_condition[(d, Xd, y)]  # 条件独立性假设
            p_post[y] = p_joint  # 分母P(X)相同，故直接存储联合概率分布即可
        return max(p_post, key=p_post.get)


if __name__ == '__main__':
    data = np.array(listCT)#分析不同贝叶斯换这个，有两个类：listCT，listT
    X_data = data[:, :-1]
    y_data = data[:, -1]
    clf = NBayes(lambda_=1)
    clf.fit(X_data, y_data)
    print(clf.p_prior, '\n', clf.p_condition,end='\n')
    pd.DataFrame(clf.p_prior,index=[range(8)]).to_csv("bayes.csv")
#关于list的后缀：此处朴素贝叶斯分析含义是在CT的情况下不同经济状况的分布，且在不同经济条件下CT的胜率（以CT为例），输出结果只看括号内第二位数字为1的情况（以CT为例，T则为2）

#卡方独立性检验
#统计各经济评级下各方胜利数
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
#合并结果
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
#卡方检验
def chi2_test(df):
    s,r=len(df.columns),len(df.index)
    x=[]
    for i in range(r):
        for j in range(s):
            nij=df.loc[df.index[i],df.columns[j]]
            mij=sum(df.loc[df.index[i]])*sum(df[df.columns[j]])/df.sum().sum()
            x.append((nij-mij)**2/mij)
    p=float(stats.chi2.sf(sum(x),(r-1)*(s-1)))
    return {'统计值':sum(x),'自由度':(r-1)*(s-1),'p值':p}
ec=chi2_test(FinalIndependenceExamData)
print(ec)
pd.DataFrame(ec,index=[0]).to_csv('Economy.csv')


#计算Person相关系数
#数据替换
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

#数据分离
classing1('ct_money')
classing1('t_money')
CTcleaned=pd.DataFrame([[i for i in dataf['ct_money']],[i for i in dataf['round_winner']]],index=['CT_money','round_winner']).T
Tcleaned=pd.DataFrame([[i for i in dataf['t_money']],[i for i in dataf['round_winner']]],index=['T_money','round_winner']).T

#统计各经济评级下各方胜利数
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
#合并结果
CTindependenceExamData=cal('ct_money')
CTindependenceExamData.index=CTindependenceExamData.iloc[:,0]
CTindependenceExamData.columns=['CT_money','win_round']
TindependenceExamData=cal('t_money')
TindependenceExamData.index=TindependenceExamData.iloc[:,0]
TindependenceExamData.columns=["T_money",'win_round']

#皮尔森系数具体计算
def pearsonRelation(data,side):
    u1,u2 = data[side].mean(),data['win_round'].mean()  # 计算均值
    std1,std2 = data[side].std(),data['win_round'].std()  # 计算标准差
    #计算相关系数所需条件
    data['(x-u1)*(y-u2)'] = (data[side] - u1) * (data['win_round'] - u2)
    data['(x-u1)**2'] = (data[side] - u1)**2
    data['(y-u2)**2'] = (data['win_round'] - u2)**2
    print(data)
    print('------')
    #计算相关系数
    r = data['(x-u1)*(y-u2)'].sum() / (np.sqrt(data['(x-u1)**2'].sum() * data['(y-u2)**2'].sum()))
    print('Pearson相关系数为：%.4f' % r)
print('T person relation:')
pearsonRelation(TindependenceExamData,'T_money')
print('CT person relation:')
pearsonRelation(CTindependenceExamData, 'CT_money')
#皮尔森相关系数说明：皮尔森相关系数仅仅说明两变量是否具有线性关系
#接上，各变量与结果的影响关系见卡方独立性检验