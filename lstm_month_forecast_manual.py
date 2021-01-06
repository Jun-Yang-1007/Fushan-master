#!/sur/bin/env python
#-*- coding: utf-8 -*-

import datetime
import pandas as pd
from database.base_env_repository import getFactors,getSites
from database.dts_env_repository import getSiteData,save_daily_forecast_data,save_month_forecast_data
from utils import dateshift
from model import  arma,lstm

import warnings
warnings.filterwarnings("ignore")


# 用于预测的历史记录

# 用于预测最大记录数
RECORD_COUNT_MAX=60

# 用于预测最小记录数
RECORD_COUNT_MIN=12

# 预测时长
RUN_LEN = 7

# 当天
#CURRENT_DATE=datetime.datetime.now()
#TODAY=CURRENT_DATE.strftime('%Y-%m-%d')
#FIRST_DAY=CURRENT_DATE.strftime('%Y-%m-01')

TODAY = '2017-03-01'
FIRST_DAY = '2017-03-01'

# 前一天
YESTERDAY = dateshift(TODAY,-1)

# 数据结束日期
END_DATETIME=YESTERDAY

# 数据开始日期
START_DATETIME = dateshift(END_DATETIME,-RECORD_COUNT_MAX,freq='M')

# 预测日期
FORECAST_START_DATETIME=TODAY
FORECAST_END_DATETIME=dateshift(TODAY,RUN_LEN-1,freq='M')

def reindex_dataframe(df,start_time=None,end_time=None,freq='MS'):
    if start_time is None:
        start_time = df.index.min()
    if end_time is None:
        end_time = df.index.max()
   
    df = df.reindex(pd.date_range(start=start_time,end=end_time,freq=freq))
 
    return df

def check_dataset(dataset):
    if dataset is None:
        print("Data Frame is None.")
        return False
    elif len(dataset) < RECORD_COUNT_MIN:
        print("Data Frame count:{0} is less than minimal required:{1}".format(len(dataset),RECORD_COUNT_MIN))
        return False
    else:
        return True 

def arma_forecast(dataset,column):
    datas = arma.forecast(dataset,days=3)
    values = datas.values
    index = datas.index
    datas = datas = pd.DataFrame(data=values,index=index,columns=[column])    
    return datas

def lstm_forecast(dataset,column):
    print("Index: ")
    index = pd.date_range(FORECAST_START_DATETIME,FORECAST_END_DATETIME,freq='MS')
    print(index)
    print("Column: ")
    print(column)
    datas = lstm.forecast(dataset,type='month')
    print("forecast data:")
    print(datas)
    datas = pd.DataFrame(data=datas,index=index,columns=[column])
    return datas



def forecast(site,factor):

    siteId=site['id']
    column=factor['column']

    df = getSiteData(siteId=siteId,factorColumn=column,start_time=START_DATETIME,end_time=END_DATETIME,type='month')

    print(df)

    if check_dataset(df):
 
        df = reindex_dataframe(df,end_time = END_DATETIME)
        df = df.interpolate('linear')

        dataset = df[column]
        
        print("data set:")
        print(dataset)
        lstm_datas = lstm_forecast(dataset,column)

        return lstm_datas     

    else:
        print("Data Frame Error.")
        return None

def save_to_database(conn):
    pass
	    
if __name__ == '__main__':
    
    print(("Start day:"+START_DATETIME+","+"End day:"+END_DATETIME))

    sites =getSites()
    factors=getFactors()    

    #sites=[{'id':'2c9394c44e95785c014e9634d8b90040','name':'test'}]
    #factors=[{"id":"xxxx","column":"F_301",'code':301}]

    for site in sites:
        for factor in factors:
            
            print("----------------------------------------------------------------")
            print("Predict Site:[{0}],Factor:{1}".format(site['id'],factor['code']))                    

            lstm_datas = forecast(site,factor)
           
    
            if lstm_datas is not None:
                #pass
                save_month_forecast_data(FIRST_DAY,site['id'],lstm_datas.values,factor['code'],model="LSTM")
            else:
                print("[LSTM]: forecast site:{0},factor:{1} failed.".format(site['id'],factor['code']))

            print("-------------------------------------")
            print("LSTM Forecast Result :")
            print(lstm_datas)
           
    
