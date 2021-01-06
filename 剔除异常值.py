import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv(r'E:\MyFpi\Project1\fujian-water-master-PY3\Data\section_1_day_data.csv',index_col=0)



anomalies = []

# Function to Detection Outlier on one-dimentional datasets.

def find_anomalies(data):
# Set upper and lower limit to 3 standard deviation
    data_std = np.std(data)
    data_mean = np.mean(data)
    anomaly_cut_off = data_std * 2
    lower_limit = data_mean - anomaly_cut_off
    upper_limit = data_mean + anomaly_cut_off
#     print(np.std(data),np.mean(data),lower_limit,upper_limit)
    for i,num in enumerate(data):
        if num > upper_limit or num < lower_limit:
            anomalies.append(num)
            #print(i,num)
            
    return None

find_anomalies(data= df ['DO'])

test3=list(df.DO)
df3=df
for outlier in anomalies:
    test3.remove(outlier)
    df3=df3[~df3.DO.isin([outlier])]
print('++++++',len(df))
print(len(df3))


anomalies.clear()

print('下一个参数------------------')




find_anomalies(data=df['水温'])

test4=list(df['水温'])
df3=df3
for outlier in anomalies:
    test4.remove(outlier)
    df3=df3[~df3.水温.isin([outlier])]
print('++++++',len(df))
print(len(df3))

anomalies.clear() 

print('下一个参数------------------')




find_anomalies(data=df['Mn'])

test4=list(df['Mn'])
df3=df3
for outlier in anomalies:
    test4.remove(outlier)
    df3=df3[~df3.Mn.isin([outlier])]
print('++++++',len(df))
print(len(df3))


#df3.to_csv('./newdata3.csv')