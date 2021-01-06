#!/sur/bin/env python
#-*- coding: utf-8 -*-
import datetime
import pandas as pd
from database.base_env_repository import getFactors, getSites
from database.dts_env_repository import getSiteData, save_daily_forecast_data, save_month_forecast_data
from database.dts_env_repository import save_hour_forecast_data
from utils import dateshift_Hour, dateshift_Month, dateshift_Day
from model import lstm
from apscheduler.schedulers.blocking import BlockingScheduler
import warnings
warnings.filterwarnings("ignore")
from wavelet import wavelet_cal
# 用于预测最大记录数
RECORD_COUNT_MAX = 1000              #60

# 用于预测最小记录数
RECORD_COUNT_MIN = 12

# 预测时长
RUN_LEN = 7
TODAY=END_DATETIME=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
TODAY="2020-10-25 10:00:00"
YESTERDAY = dateshift_Hour(TODAY,-1)

# 数据开始日期
START_DATETIME = dateshift_Hour(END_DATETIME,-RECORD_COUNT_MAX)
# 数据结束日期
END_DATETIME=YESTERDAY

def make_print_to_file(path='./'):

    import os
    import sys
    import datetime

    class Logger(object):
        def __init__(self, filename="Default.log", path="./"):
            self.terminal = sys.stdout
            self.log = open(os.path.join(path, filename), "a", encoding='utf8', )

        def write(self, message):
            self.terminal.write(message)
            self.log.write(message)

        def flush(self):
            pass
    fileName = datetime.datetime.now().strftime('day'+'%Y-%m-%d')
    sys.stdout = Logger(fileName + '.hourlylog', path=path)

def main(sites,factors):
    global TODAY, END_DATETIME, FORECAST_END_DATETIME, FORECAST_START_DATETIME
    TODAY=END_DATETIME=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    #TODAY=END_DATETIME='2020-09-01 00:00:00'
    FORECAST_START_DATETIME=TODAY
    FORECAST_END_DATETIME=dateshift_Hour(TODAY,RUN_LEN  -1)
    print(("Start day:" + START_DATETIME + "," + "End day:" + END_DATETIME))
    make_print_to_file(path='./')  # 输出日志
    for site in sites:
        for factor in factors:

            print("----------------------------------------------------------------")
            print(("Predict Site:[{0}],Factor:{1}".format(site['id'], factor['code'])))

            lstm_datas = forecast(site, factor)

            if lstm_datas is not None:
                save_hour_forecast_data(FORECAST_START_DATETIME, site['id'], lstm_datas.values, factor['code'],model="LSTM")
            else:
                print(("[LSTM]: forecast site:{0},factor:{1} failed.".format(site['id'], factor['code'])))

            print("-------------------------------------")
            print("LSTM Forecast Result :")
            print(lstm_datas)
# 当天



def reindex_dataframe(df, start_time=None, end_time=None, freq='H'):
    if start_time is None:
        start_time = df.index.min()
    if end_time is None:
        end_time = df.index.max()
    '''
    when the id means {'id': '2c90827271c3757f0171c4b793e70006', 'name': '釜山水库固定站'}, showing data from every 4 days
    the freq  must be freq = 4H
    '''
    df = df.reindex(pd.date_range(start=start_time, end=end_time, freq=freq))
    return df


def check_dataset(dataset):
    if dataset is None:
        print("Data Frame is None.")
        return False
    elif len(dataset) < RECORD_COUNT_MIN:
        print(("Data Frame count:{0} is less than minimal required:{1}".format(len(dataset), RECORD_COUNT_MIN)))
        return False  # 两个return false
    else:
        return True


def lstm_forecast(dataset, column):
    datas = lstm.forecast(dataset, type='daily')

    '''
    when the id means {'id': '2c90827271c3757f0171c4b793e70006', 'name': '釜山水库固定站'}, showing data from every 4 days
    the freq  must be freq = 4H
    '''
    if siteId == '2c90827271c3757f0171c4b793e70006':
        forecast_end_datetime2 = dateshift_Hour(TODAY, RUN_LEN * 4 - 1)
        index = pd.date_range(FORECAST_START_DATETIME, forecast_end_datetime2, freq='4H')
    else:
        index = pd.date_range(FORECAST_START_DATETIME, FORECAST_END_DATETIME, freq='H')
    datas = pd.DataFrame(data=datas, index=index, columns=[column])
    return datas


def forecast(site, factor):
    global siteId
    siteId = site['id']
    print('siteid是什么',siteId )
    column = factor['column']
    df = getSiteData(siteId=siteId, factorColumn=column, start_time=START_DATETIME, end_time=END_DATETIME, type='hour')
    print(df)
   # df = wavelet_cal(df)
    print('****************')

    if check_dataset(df):
        if siteId =='2c90827271c3757f0171c4b793e70006':
            df = reindex_dataframe(df, start_time=None, end_time=None, freq='4H')
        else:

            df = reindex_dataframe(df, end_time=END_DATETIME)
     #   df = df.interpolate('linear')

        dataset = df[column]

        lstm_datas = lstm_forecast(dataset, column)
        print('釜山水库固定站预测结果', lstm_datas)
        return lstm_datas

    else:
        print("Data Frame Error.")
        return None
    # 需要用到 siteID
    return siteId


def save_to_database(conn):
    pass





if __name__ == '__main__':
    scheduler = BlockingScheduler()

    print(("Start day:" + START_DATETIME + "," + "End day:" + END_DATETIME))

    sites = getSites()
    factors = getFactors()
    print(sites)
    # sites=[{'id':'2c9394c44e95785c014e9634d8b90040','name':'test'}]
    main(sites,factors)
    # scheduler.add_job(main, 'interval', seconds=60, args=[sites, factors])
    #scheduler.add_job(main, 'interval', seconds=3600, args=[sites, factors])
    # scheduler.add_job(main, 'cron', hour='0-23', minute='00',  args=[sites, factors],misfire_grace_time = 2400)
    #TODAY=END_DATETIME=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    # #
   # scheduler.start()






