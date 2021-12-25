#coding=gbk
import pandas as pd
import numpy as np
from collections import defaultdict,Counter
data=pd.read_csv('BDAM\csgo_round_snapshots.csv')
data_df=pd.DataFrame(data)
data_df=data_df.loc[:,['map','round_winner']]
classification1=np.unique(data_df['map'])#��ȡ��ͼ������
print(classification1)
for i in classification1:#�滻��ͼΪ����
    for k in range((len(data_df))):
        if data_df.loc[k,'map']==i:
            a1=np.where(classification1==i)#��ȡ��ͼλ��������ڼ�λ
            b1=a1[0][0]#��ȡ�������ж�Ӧ����
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
#��dataframeת����array
list1=[]
list2=[]
for i in range(len(data_df)):
    list1.append(data_df.loc[i,'map'])   
for i in range(len(data_df)):
    list2.append(data_df.loc[i,'round_winner'])
list3=list(zip(list2,list1))
#���ر�Ҷ˹���ʷ���
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
    data = np.array(list3)
    X_data = data[:, :-1]
    y_data = data[:, -1]
    clf = NBayes(lambda_=1)
    clf.fit(X_data, y_data)
    print(clf.p_prior, '\n', clf.p_condition)
    pd.DataFrame(clf.p_condition,index=[0]).T.to_csv('Map.csv')