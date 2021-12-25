#coding=gbk
#��Ҫ�����룬���˻��ò���
import pandas as pd
import numpy as np
import re
from collections import defaultdict,Counter
import sys
sys.setrecursionlimit(1000000)
import scipy.stats as stats
def containVarInString(containVar,stringVar):
        if isinstance(stringVar, str):
            if stringVar.find(containVar)!=-1:
                return True
            else:
                return False
        else:
            return False
data=pd.read_csv('BDAM\csgo_round_snapshots.csv')#�ܴ��������ļ��������·�����������ļ�
data_df=pd.DataFrame(data)
col=list(data_df.columns.values)#��ȡ������
str1=','.join(col)#��������ת���ַ����ԣ�Ϊ�ָ��
class1=re.findall('...weapon.*', str1)#������ʽƥ����߼�װ��

#������ʽ���س���Ϊ1��list�����°�װ�������������lsit
str2=''.join(class1)
classification=str2.split(',')

#ȥ��classification�����ڵ��������
for i in range(len(classification)):
    e=containVarInString('grenade', classification[i])
    if e:
        classification[i]=0

#ֻ����ԭ��������װ�����������غ�ʤ��
for i in range(classification.count(0)):
    del classification[-2]
ClassifiedData=np.random.rand(len(data_df)).reshape(len(data_df),1)
ClassifiedData=pd.DataFrame(ClassifiedData,columns=['A'])
for i in range(len(classification)):
    k=classification[-i-1]
    a=pd.Series(data_df.loc[:,k])
    ClassifiedData.insert(1, k, a)
ClassifiedData=ClassifiedData.drop(columns='A')
score=pd.read_csv(r'BDAM\Project\equipment\temporary.csv')#Ҫ�ܴ�������·���������·��
score=pd.DataFrame(score)
score.index=classification[0:68]

#����װ������
CTrank=[]
Trank=[]
for i in range(len(ClassifiedData)):
    CTaddup=0
    Taddup=0
    for k in classification:
        if k != 'round_winner':
            if k.find('ct')>=0:
                result=float(score.loc[k,'Score']*ClassifiedData.loc[i,k])
                CTaddup+=result
            else:
                result=float(score.loc[k,'Score']*ClassifiedData.loc[i,k])
                Taddup+=result
        else:
            break
    CTrank.append(CTaddup)
    Trank.append(Taddup)
CTrank=pd.Series(CTrank)
ClassifiedData.insert(0,'CTrank',CTrank)
Trank=pd.Series(Trank)
ClassifiedData.insert(0,'Trank',Trank)
del classification[-1]
ClassifiedData.drop(columns=[i for i in classification],inplace=True)

#ʤ�������滻
for i in range(len(ClassifiedData)):
    if ClassifiedData.loc[i,'round_winner']=='CT':
        ClassifiedData.loc[i,'round_winner']=1
    else:
        ClassifiedData.loc[i,'round_winner']=2

#�����������
for i in range(len(ClassifiedData)):
    if ClassifiedData.loc[i,'Trank']<93:
        ClassifiedData.drop([i],axis=0,inplace=True)
    elif ClassifiedData.loc[i,'CTrank']<157:
        ClassifiedData.drop([i],axis=0,inplace=True)
ClassifiedData.index=range(len(ClassifiedData))

#�滻װ�����֣�����װ���ȼ�����150��Ϊ����
a=[93,157]
def equipmentclass(dataset,side):
    if side=='CTank':
        minimum=a[1]
    else:
        minimum=a[0]
    for i in range(len(ClassifiedData)):
        if minimum<=ClassifiedData.loc[i,side]<=(minimum+150):
            ClassifiedData.loc[i,side]='F'
        elif (minimum+150)<ClassifiedData.loc[i,side]<=(minimum+300):
            ClassifiedData.loc[i,side]='E'
        elif (minimum+300)<ClassifiedData.loc[i,side]<=(minimum+450):
            ClassifiedData.loc[i,side]='D'
        elif (minimum+450)<ClassifiedData.loc[i,side]<=(minimum+600):
            ClassifiedData.loc[i,side]='C'
        elif (minimum+600)<ClassifiedData.loc[i,side]<=(minimum+750):
            ClassifiedData.loc[i,side]='B'
        elif (minimum+750)<ClassifiedData.loc[i,side]:
            ClassifiedData.loc[i,side]='A'
    return dataset
ClassifiedData=equipmentclass(ClassifiedData,'CTrank')
ClassifiedData=equipmentclass(ClassifiedData,'Trank')

#��dataframeת����array
list1=[]
list2=[]
for i in range(len(ClassifiedData)):
    list1.append(ClassifiedData.loc[i,'Trank'])   
for i in range(len(ClassifiedData)):
    list2.append(ClassifiedData.loc[i,'round_winner'])
listT=list(zip(list2,list1))
list1=[]
list2=[]
for i in range(len(ClassifiedData)):
    list1.append(ClassifiedData.loc[i,'CTrank'])   
for i in range(len(ClassifiedData)):
    list2.append(ClassifiedData.loc[i,'round_winner'])
listCT=list(zip(list2,list1))
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
    print(clf.p_prior, '\n', clf.p_condition)
    pd.DataFrame(clf.p_condition,index=[0]).T.to_csv('Equipment.csv')

#���������Լ���
#ͳ�Ƹ�װ�������¸���ʤ����
def cal(side):
    dataset=ClassifiedData.groupby([side])
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
CTindependenceExamData=cal('CTrank')
CTindependenceExamData.index=CTindependenceExamData.iloc[:,0]
CTindependenceExamData.drop([0],axis=1,inplace=True)
CTindependenceExamData.columns=['CT']
CTindependenceExamData=pd.Series([i for i in CTindependenceExamData['CT']],index=['A','B','C','D','E','F'])
TindependenceExamData=cal('Trank')
TindependenceExamData.index=TindependenceExamData.iloc[:,0]
TindependenceExamData.drop([0],axis=1,inplace=True)
TindependenceExamData.columns=["T"]
TindependenceExamData=pd.Series([i for i in TindependenceExamData['T']],index=['A','B','C','D','E','F'])
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
print(chi2_test(FinalIndependenceExamData))