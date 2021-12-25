#coding=gbk
import pandas as pd
import numpy as np
from collections import defaultdict,Counter
data=pd.read_csv('BDAM\csgo_round_snapshots.csv')
data_df=pd.DataFrame(data)
data_df=data_df.loc[:,['map','round_winner']]
classification1=np.unique(data_df['map'])#提取地图特征组
print(classification1)
for i in classification1:#替换地图为数据
    for k in range((len(data_df))):
        if data_df.loc[k,'map']==i:
            a1=np.where(classification1==i)#提取地图位于特征组第几位
            b1=a1[0][0]#提取特征组中对应索引
            data_df.loc[k,'map']=b1
classification2=np.unique(data_df['round_winner'])
for i in classification2:
    for k in range(len(data_df)):
        if data_df.loc[k,'round_winner']==i:
            a2=np.where(classification2==i)
            b2=a2[0][0]
            data_df.loc[k,'round_winner']=b2
print(classification1)
print(classification2)       
#将dataframe转换成array
list1=[]
list2=[]
for i in range(len(data_df)):
    list1.append(data_df.loc[i,'map'])   
for i in range(len(data_df)):
    list2.append(data_df.loc[i,'round_winner'])
list3=list(zip(list2,list1))
#朴素贝叶斯概率分析
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
    data = np.array(list3)
    X_data = data[:, :-1]
    y_data = data[:, -1]
    clf = NBayes(lambda_=1)
    clf.fit(X_data, y_data)
    print(clf.p_prior, '\n', clf.p_condition)
    pd.DataFrame(clf.p_condition,index=[0]).T.to_csv('Map.csv')