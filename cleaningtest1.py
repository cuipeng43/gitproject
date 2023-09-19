import pandas as pd
import random
missing_value = ['n/a', 'na', '--']
df = pd.read_csv('data.csv', na_values=missing_value)#读取csv文件 符合条件的值设置为NaN
print(df)
df['PID'] = df['PID'].fillna(df['PID'].interpolate())#以NaN值前后的平均值填充PID列NaN值
df['ST_NUM'].fillna(method='ffill', inplace=True)#以NaN值前面的值填充
for x in df.index:
    if df.loc[x,'OWN_OCCUPIED']!='Y' and df.loc[x,'OWN_OCCUPIED']!='N':#OWN_OCCUPIED列中非Y非N值随机填充Y或者N
        df.loc[x,'OWN_OCCUPIED']=random.choice(['Y','N'])
df['NUM_BEDROOMS'].fillna(value=df['NUM_BEDROOMS'].mode()[0],inplace=True)#NUM_BEDROOMS列中的NaN以众数填充 众数不唯一 [0]代表第一个众数
for y in df.index:
    try:
        df.loc[y,'NUM_BATH']=float(df.loc[y,'NUM_BATH']) #try测试类型转换是否存在异常，出现异常删除该行
    except:
        df.drop(index=y,axis=0,inplace=True)
df['NUM_BATH'].fillna(df['NUM_BATH'].mode()[0],inplace=True)#众数填充NaN
df['SQ_FT'].fillna(df['SQ_FT'].mean(),inplace=True)#平均数填充NaN
df=df.astype({'PID':'int','ST_NUM':'int','NUM_BEDROOMS':'int','SQ_FT':'int'})#float类型转换为int类型
df.reset_index(drop=True,inplace=True)

