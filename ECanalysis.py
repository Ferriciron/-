#coding=utf-8
import pandas as pd
import numpy as np
import matplotlib as plt
data=pd.read_csv(r'BDAM\csgo_round_snapshots.csv')#跑代码的时候改这个就好，无需其他设置
dataf=pd.DataFrame(data)
dataf=dataf.loc[:,['ct_money','round_winner']]
#提取相关信息
for i in range(len(dataf)):
    if dataf.loc[i,'round_winner']=='CT':
        dataf.loc[i,'round_winner']=1
    else:
        dataf.loc[i,'round_winner']=2
#经济分类
bad=[]
normal=[]
good=[]
excellent=[]
for i in range(len(dataf)):
    if int(dataf.loc[i,'ct_money'])<1500:
        bad.append(dataf.loc[i,:])
    elif 1500<=int(dataf.loc[i,'ct_money'])<3500:
        normal.append(dataf.loc[i,:])
    elif 3500<=int(dataf.loc[i,'ct_money'])<6000:
        good.append(dataf.loc[i,:])
    elif 6000<=int(dataf.loc[i,'ct_money']):
        excellent.append(dataf.loc[i,:])
#转换成Dataframe
bad=pd.DataFrame(bad)
bad.index=range(len(bad))
normal=pd.DataFrame(normal)
normal.index=range(len(normal))
good=pd.DataFrame(good)
good.index=range(len(good))
excellent=pd.DataFrame(excellent)
excellent.index=range(len(excellent))
#相关性分析
def winningnumber(dataset):
    ctwinning,twinning=0,0
    for i in range(len(dataset)):
        if dataset.loc[i,'round_winner']==1:
            ctwinning += 1
        else:
            twinning += 1
    return ctwinning,twinning

def pro(dataset):
    ctwin,twin=winningnumber(dataset)
    ct_probability=ctwin/(ctwin+twin)
    t_probability=twin/(twin+ctwin)
    return ct_probability,t_probability

def winninglinear(dataset):
    ctwin,twin=winningnumber(dataset)
    ctpro,tpro=pro(dataset)
    data=[{'win':ctwin,"Probability":ctpro},{'win':twin,'Probability':tpro}]
    data=pd.DataFrame(data)
    u1,u2 = data['win'].mean(),data['Probability'].mean()  # 计算均值
    std1,std2 = data['win'].std(),data['Probability'].std()  # 计算标准差
    #计算相关系数所需条件
    data['(x-u1)*(y-u2)'] = (data['win'] - u1) * (data['Probability'] - u2)
    data['(x-u1)**2'] = (data['win'] - u1)**2
    data['(y-u2)**2'] = (data['Probability'] - u2)**2
    print(data.head())
    print('------')
    #计算相关系数
    r = data['(x-u1)*(y-u2)'].sum() / (np.sqrt(data['(x-u1)**2'].sum() * data['(y-u2)**2'].sum()))
    print('Pearson相关系数为：%.4f' % r)
ct_probad,t_probad=pro(bad)
ct_pronormal,t_pronormal=pro(normal)
ct_progood,t_progood=pro(good)
ct_proexcellent,t_proexcellent=pro(excellent)
ct_bad,t_bad=winningnumber(bad)
ct_normal,t_normal=winningnumber(normal)
ct_good,t_good=winningnumber(good)
ct_execellent,t_excellent=winningnumber(excellent)
t_semidata=[{'Economy level':t_bad,'Probability':t_probad},{'Economy level':t_normal,'Probability':t_pronormal},{'Economy level':t_good,'Probability':t_progood},{'Economy level':t_excellent,'Probability':t_proexcellent}]
ct_semidata=[{'Economy level':ct_bad,'Probability':ct_probad},{'Economy level':ct_normal,'Probability':ct_pronormal},{'Economy level':ct_good,'Probability':ct_progood},{'Economy level':ct_execellent,'Probability':ct_proexcellent}]
t_semidata=pd.DataFrame(t_semidata)
ct_semidata=pd.DataFrame(ct_semidata)
data=pd.DataFrame(data)
#总相关性分析
def wholelinear(data):
    u1,u2 = data['Economy level'].mean(),data['Probability'].mean()  # 计算均值
    std1,std2 = data['Economy level'].std(),data['Probability'].std()  # 计算标准差
    #计算相关系数所需条件
    data['(x-u1)*(y-u2)'] = (data['Economy level'] - u1) * (data['Probability'] - u2)
    data['(x-u1)**2'] = (data['Economy level'] - u1)**2
    data['(y-u2)**2'] = (data['Probability'] - u2)**2
    print(data.head())
    print('------')
    #计算相关系数
    r = data['(x-u1)*(y-u2)'].sum() / (np.sqrt(data['(x-u1)**2'].sum() * data['(y-u2)**2'].sum()))
    print('Pearson相关系数为：%.4f' % r)
print('whole analysis of t')
print('--------------------------------------------------')
wholelinear(t_semidata)
print('whole analysis of ct')
print('--------------------------------------------------')
wholelinear(ct_semidata)
print('以下数据均为先ct后t')
print('\n')
print('analysis of bad economy')
print('--------------------------------------------------')
winninglinear(bad)
print('analysis of normal economy')
print('--------------------------------------------------')
winninglinear(normal)
print('analysis of good economy')
print('--------------------------------------------------')
winninglinear(good)
print('analysis of excellent economy')
print('--------------------------------------------------')
winninglinear(excellent)