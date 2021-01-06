#!/sur/bin/env python
#-*- coding: utf-8 -*-

import datetime
import pandas as pd
from database.base_env_repository import getFactors,getSites
from database.dts_env_repository import getSiteData,save_daily_forecast_data
from utils import dateshift
from model import  arma,lstm

import warnings
warnings.filterwarnings("ignore")


# 用于预测的历史记录

# 用于预测最大记录数
RECORD_COUNT_MAX=365                       #60

# 用于预测最小记录数
RECORD_COUNT_MIN=12

# 预测时长
RUN_LEN =365   #4

# 当天
#TODAY=datetime.datetime.now().strftime('%Y-%m-%d')
def getTime(date):
    #TODAY = date    #yuuanlai
    TODAY = '2019-07-31'

    # 前一天
    YESTERDAY = dateshift(TODAY,-1)

    # 数据结束日期
    END_DATETIME=YESTERDAY

    # 数据开始日期
    START_DATETIME = dateshift(END_DATETIME,-RECORD_COUNT_MAX)          #开始日期是昨天-最长预测记录长度  
                                                                        #要测2020.05.06的，测5天。那么就从2020.05.01开始

    # 预测日期
    FORECAST_START_DATETIME=TODAY
    FORECAST_END_DATETIME=dateshift(TODAY,RUN_LEN-1)                   #预测截止日期是今天+预测长度，即到预测那天为止
    return START_DATETIME,END_DATETIME,FORECAST_START_DATETIME,FORECAST_END_DATETIME
    
def reindex_dataframe(df,start_time=None,end_time=None,freq='1D'):
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


def lstm_forecast(dataset,column,start_time,end_time):
    datas = lstm.forecast(dataset)
    index = pd.date_range(start_time,end_time,freq='D')
    datas = pd.DataFrame(data=datas,index=index,columns=[column])
    return datas



def forecast(site,factor,start_time,end_time,forecast_start_time,forecast_end_time):

    siteId=site['id']
    column=factor['column']

    df = getSiteData(siteId=siteId,factorColumn=column,start_time=start_time,end_time=end_time)

    # print df

    if check_dataset(df):
 
        df = reindex_dataframe(df,end_time = end_time)
        df = df.interpolate('linear')

        dataset = df[column]

        lstm_datas = lstm_forecast(dataset,column,forecast_start_time,forecast_end_time)

        return lstm_datas     

    else:
        print("Data Frame Error.")
        return None

def save_to_database(conn):
    pass
	    
if __name__ == '__main__':
    
    sites =getSites()
    factors=getFactors()    

    #sites=[{'id':'2c9394c44e95785c014e9634d8b90040','name':'test'}]
    #factors=[{"id":"xxxx","column":"F_301",'code':301}]
    
    #BEGIN_TIME = '2017-07-15'
    BEGIN_TIME = '2020-07-20'
    for x in range(0,15):
        today = dateshift(BEGIN_TIME,x)

        start_time,end_time,forecast_start_time,forecast_end_time = getTime(today)

        print(("Start day:"+start_time+","+"End day:"+end_time))
       
        for site in sites:
            for factor in factors:
            
                print("----------------------------------------------------------------")
                print("Predict Site:[{0}],Factor:{1}".format(site['id'],factor['code']))                    

                lstm_datas = forecast(site,factor,start_time,end_time,forecast_start_time,forecast_end_time)
           
    
                if lstm_datas is not None:
                    save_daily_forecast_data(forecast_start_time,site['id'],lstm_datas.values,factor['code'],model="LSTM")
                else:
                    print("[LSTM]: forecast site:{0},factor:{1} failed.".format(site['id'],factor['code']))

                print("-------------------------------------")
                print("LSTM Forecast Result :")
                print(lstm_datas)
           
    
