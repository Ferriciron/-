import pandas as pd
import numpy as np
def ctdroping(dataset):
    tempdataset=dataset.loc[:,['ct_money','round_winner']]
    return tempdataset
def tdroping(dataset):
    tempdataset=dataset.loc[:,['t_money','round_winner']]
    return tempdataset
dt=pd.read_csv(r'C:\Users\1\Desktop\csgo_round_snapshots.csv')
dt1=dt.loc[:,['map','t_money','round_winner']]
dt1.to_csv('terrorist_economy.csv')
dt2=dt.loc[:,['map','ct_money','round_winner']]
dt2.to_csv('ct_economy.csv')
dtt=pd.read_csv(r'terrorist_economy.csv')
dtct=pd.read_csv(r'ct_economy.csv')
dfct=pd.DataFrame(dtct)
dft=pd.DataFrame(dtt)
df=pd.DataFrame(dt)
classifiction=df['map'].unique()
for tempclassification in classifiction:
    tempdata=dfct[dfct['map'].isin([tempclassification])]
    exec('dfct%s=tempdata'%tempclassification)
for tempclassification in classifiction:
    tempdata=dft[dft['map'].isin([tempclassification])]
    exec('dft%s=tempdata'%tempclassification)
dfctde_dust2=ctdroping(dfctde_dust2)
dfctde_mirage=ctdroping(dfctde_mirage)
dfctde_nuke=ctdroping(dfctde_nuke)
dfctde_inferno=ctdroping(dfctde_inferno)
dfctde_overpass=ctdroping(dfctde_overpass)
dfctde_train=ctdroping(dfctde_train)
dfctde_cache=ctdroping(dfctde_cache)
dfctde_vertigo=ctdroping(dfctde_vertigo)
dftde_dust2=tdroping(dftde_dust2)
dftde_mirage=tdroping(dftde_mirage)
dftde_nuke=tdroping(dftde_nuke)
dftde_inferno=tdroping(dftde_inferno)
dftde_overpass=tdroping(dftde_overpass)
dftde_train=tdroping(dftde_train)
dftde_cache=tdroping(dftde_cache)
dftde_vertigo=tdroping(dftde_vertigo)
dfctde_dust2.to_csv('counterterrorist_dust2.csv')
dfctde_mirage.to_csv('counterterrorist_mirage.csv')
dfctde_nuke.to_csv('counterterrorist_nuke.csv')
dfctde_inferno.to_csv('counterterrorist_inferno.csv')
dfctde_overpass.to_csv('counterterrorist_overpass.csv')
dfctde_train.to_csv('counterterrorist_train.csv')
dfctde_cache.to_csv('counterterrorist_cache.csv')
dfctde_vertigo.to_csv('counterterrorist_vertigo.csv')
dftde_dust2.to_csv('terrorist_dust2.csv')
dftde_mirage.to_csv('terrorist_mirage.csv')
dftde_nuke.to_csv('terrorist_nuke.csv')
dftde_inferno.to_csv('terrorist_inferno.csv')
dftde_overpass.to_csv('terrorist_overpass.csv')
dftde_train.to_csv('terrorist_train.csv')
dftde_cache.to_csv('terrorist_cache.csv')
dftde_vertigo.to_csv('terrorist_vertigo.csv')