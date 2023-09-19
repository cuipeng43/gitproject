import pandas as pd
import time
df = pd.read_csv('baltic_ais_with_anomalies.csv')
for x in df.index:
    if df.loc[x,'MMSI']<100000000:
        df.drop(index=x,axis=0,inplace=True) #MMSI位数少于9 删除该行
for y in df.index:
    timearray=time.localtime(df.loc[y,'TIME'])  #将timestamp按当地时区转换
    df.loc[y, 'TIME']=time.strftime("%Y-%m-%d %H:%M:%S", timearray) #将timestamp按照 xxxx-xx-xx xx:xx:xx的格式转换
df.reset_index(drop=True,inplace=True) #重置index
print(df)