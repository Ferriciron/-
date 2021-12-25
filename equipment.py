#coding=gbk
#不要动代码，动了会变得不幸
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
data=pd.read_csv('BDAM\csgo_round_snapshots.csv')#跑代码改这个文件，用相对路径，填入主文件
data_df=pd.DataFrame(data)
col=list(data_df.columns.values)#提取列索引
str1=','.join(col)#将列索引转成字符串以，为分割点
class1=re.findall('...weapon.*', str1)#正则表达式匹配道具及装备

#正则表达式返回长度为1的list，以下按装备道具重新组成lsit
str2=''.join(class1)
classification=str2.split(',')

#去除classification中属于道具类的项
for i in range(len(classification)):
    e=containVarInString('grenade', classification[i])
    if e:
        classification[i]=0

#只留下原有数据中装备相关数据与回合胜者
for i in range(classification.count(0)):
    del classification[-2]
ClassifiedData=np.random.rand(len(data_df)).reshape(len(data_df),1)
ClassifiedData=pd.DataFrame(ClassifiedData,columns=['A'])
for i in range(len(classification)):
    k=classification[-i-1]
    a=pd.Series(data_df.loc[:,k])
    ClassifiedData.insert(1, k, a)
ClassifiedData=ClassifiedData.drop(columns='A')
score=pd.read_csv(r'BDAM\Project\equipment\temporary.csv')#要跑代码改这个路径，用相对路径
score=pd.DataFrame(score)
score.index=classification[0:68]

#计算装备评分
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

#胜方数字替换
for i in range(len(ClassifiedData)):
    if ClassifiedData.loc[i,'round_winner']=='CT':
        ClassifiedData.loc[i,'round_winner']=1
    else:
        ClassifiedData.loc[i,'round_winner']=2

#清除无用数据
for i in range(len(ClassifiedData)):
    if ClassifiedData.loc[i,'Trank']<93:
        ClassifiedData.drop([i],axis=0,inplace=True)
    elif ClassifiedData.loc[i,'CTrank']<157:
        ClassifiedData.drop([i],axis=0,inplace=True)
ClassifiedData.index=range(len(ClassifiedData))

#替换装备评分，换成装备等级，以150分为步长
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

#将dataframe转换成array
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
    print(clf.p_prior, '\n', clf.p_condition)
    pd.DataFrame(clf.p_condition,index=[0]).T.to_csv('Equipment.csv')

#卡方独立性检验
#统计各装备评级下各方胜利数
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
#合并结果
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
print(chi2_test(FinalIndependenceExamData))