import pandas as pd
import numpy as np
def counting(array):
    a=1
    for i in range(len(array)):
        if array[i]==1:
            a += 1
    return a
def winpercentage(array):
    a=counting(array)/len(array)
    return a
#读取数据   
data=pd.read_csv(r'C:\Users\1\Desktop\csgo_round_snapshots.csv')
df=pd.DataFrame(data)
#对地图进行数字替换
for i in range(int(len(df))):
    if df.loc[i,'map'] == 'de_dust2':
        df.loc[i,'map']=1
    elif df.loc[i,'map'] == 'de_mirage':
        df.loc[i,'map']=2
    elif df.loc[i,'map'] == 'de_nuke':
        df.loc[i,'map']=3
    elif df.loc[i,'map'] == 'de_inferno':
        df.loc[i,'map']=4
    elif df.loc[i,'map'] == 'de_overpass':
        df.loc[i,'map']=5
    elif df.loc[i,'map'] == 'de_vertigo':
        df.loc[i,'map']=6
    elif df.loc[i,'map'] == 'de_train':
        df.loc[i,'map']=7
    elif df.loc[i,'map'] == 'de_cache':
        df.loc[i,'map']=8
df1=df.loc[:,['map','round_winner']]
print(type(df1))
for i in range(len(df1)):
    if df1.loc[i,'round_winner'] == 'CT':
        df1.loc[i,'round_winner'] = 1
    else:
        df1.loc[i,'round_winner']=2
dust2=[]
mirage=[]
nuke=[]
inferno=[]
overpass=[]
vertigo=[]
train=[]
cache=[]
for i in range(len(df1)):
    if df1.loc[i,'map']==1:
        dust2.append(df1.loc[i,'round_winner'])
    elif df1.loc[i,'map']==2:
            mirage.append(df1.loc[i,'round_winner'])
    elif df1.loc[i,'map']==3:
            nuke.append(df1.loc[i,'round_winner'])
    elif df1.loc[i,'map']==4:
            inferno.append(df1.loc[i,'round_winner'])
    elif df1.loc[i,'map']==5:
            overpass.append(df1.loc[i,'round_winner']) 
    elif df1.loc[i,'map']==6:
            vertigo.append(df1.loc[i,'round_winner'])
    elif df1.loc[i,'map']==7:
            train.append(df1.loc[i,'round_winner'])  
    elif df1.loc[i,'map']==8:
            cache.append(df1.loc[i,'round_winner'])
a=[winpercentage(dust2),winpercentage(mirage),winpercentage(nuke),winpercentage(inferno),winpercentage(overpass),winpercentage(vertigo),winpercentage(train),winpercentage(cache)]
b=['dust2','mirage','nuke','inferno','overpass','vetigo','train','cache']
c=[]
for i in range(len(a)):
    d=1-a[i]
    c.append(d)
result=pd.DataFrame([b,a,c],index=['Map','Probability of Winning in CT','Probability of Winning in T'])
result.to_csv('finaldata.csv')