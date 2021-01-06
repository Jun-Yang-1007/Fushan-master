#!/sur/bin/env python
# -*- coding: utf-8 -*-
import datetime
import pandas as pd
from database.base_env_repository import getFactors, getSites
from database.dts_env_repository import getSiteData, save_daily_forecast_data, save_month_forecast_data
from database.dts_env_repository import save_hour_forecast_data
from utils import dateshift_Hour, dateshift_Month, dateshift_Day
from model import lstm

from apscheduler.schedulers.blocking import BlockingScheduler
from database.missing_value_filling import set_missing
import warnings

warnings.filterwarnings("ignore")

# 用于预测的历史记录

# 用于预测最大记录数
RECORD_COUNT_MAX = 60

# 用于预测最小记录数
RECORD_COUNT_MIN = 12

# 预测时长
RUN_LEN = 7

# TODAY = END_DATETIME = datetime.datetime.now().strftime("%Y-%m-%d")
TODAY = END_DATETIME = '2020-07-27'
YESTERDAY = dateshift_Day(TODAY, -1)

# 数据开始日期
START_DATETIME = dateshift_Day(END_DATETIME, -RECORD_COUNT_MAX)
# 数据结束日期
END_DATETIME = YESTERDAY


def main(sites, factors):
    global TODAY, END_DATETIME, FORECAST_END_DATETIME, FORECAST_START_DATETIME
    # TODAY = END_DATETIME = datetime.datetime.now().strftime("%Y-%m-%d")
    TODAY = END_DATETIME = '2020-08-21'
    # TODAY=END_DATETIME='2020-09-01'
    FORECAST_START_DATETIME = TODAY
    FORECAST_END_DATETIME = dateshift_Day(TODAY, RUN_LEN - 1)
    print(("Start day:" + START_DATETIME + "," + "End day:" + END_DATETIME))
    for site in sites:
        for factor in factors:
            print("----------------------------------------------------------------")
            print(("Predict Site:[{0}],Factor:{1}".format(site['id'], factor['code'])))
            lstm_datas = forecast(site, factor)

            if lstm_datas is not None:
                save_daily_forecast_data(FORECAST_START_DATETIME, site['id'], lstm_datas.values, factor['code'],
                                         model="LSTM")
            else:
                print(("[LSTM]: forecast site:{0},factor:{1} failed.".format(site['id'], factor['code'])))

            print("-------------------------------------")
            print("LSTM Forecast Result :")
            print(lstm_datas)


# 当天

# TODAY=datetime.datetime.now().strftime("%Y-%m-%d")
# print(TODAY)
# TODAY = '2020-09-01'

# 前一天
# YESTERDAY = dateshift_Day(TODAY,-1)


# 数据结束日期
# END_DATETIME=YESTERDAY
# END_DATETIME=TODAY
# 数据开始日期
# START_DATETIME = dateshift(END_DATETIME,-RECORD_COUNT_MAX)
START_DATETIME = '2020-08-28'


# START_DATETIME='2020-08-01'

# 预测日期
# FORECAST_START_DATETIME=TODAY
# FORECAST_END_DATETIME=dateshift_Day(TODAY,RUN_LEN-1)

def reindex_dataframe(df, start_time=None, end_time=None, freq='1D'):
    if start_time is None:
        start_time = df.index.min()
    if end_time is None:
        end_time = df.index.max()
    df = df.reindex(pd.date_range(start=start_time, end=end_time, freq=freq))

    return df


def check_dataset(dataset):
    if dataset is None:
        print("Data Frame is None.")
        return False
    elif len(dataset) < RECORD_COUNT_MIN:
        print(("Data Frame count:{0} is less than minimal required:{1}".format(len(dataset), RECORD_COUNT_MIN)))
        return False
    else:
        return True


def lstm_forecast(dataset, column):
    datas = lstm.forecast(dataset, type='daily')
    index = pd.date_range(FORECAST_START_DATETIME, FORECAST_END_DATETIME, freq='D')
    datas = pd.DataFrame(data=datas, index=index, columns=[column])
    return datas


def forecast(site, factor):
    siteId = site['id']
    column = factor['column']
    df = getSiteData(siteId=siteId, factorColumn=column, start_time=START_DATETIME, end_time=END_DATETIME, type='daily')
    print(df)

    if check_dataset(df):
        df = reindex_dataframe(df, end_time=END_DATETIME)
        df = df.interpolate('linear')

        dataset = df[column]

        lstm_datas = lstm_forecast(dataset, column)

        return lstm_datas

    else:
        print("Data Frame Error.")
        return None


def save_to_database(conn):
    pass


if __name__ == '__main__':
    scheduler = BlockingScheduler()

    # print(("Start day:"+START_DATETIME+","+"End day:"+END_DATETIME))

    sites = getSites()
    factors = getFactors()
    print(sites)
    # sites=[{'id':'2c9394c44e95785c014e9634d8b90040','name':'test'}]
    # factors=[{"id":"xxxx","column":"F_301",'code':301}]
    scheduler.add_job(main, 'interval', seconds=10, args=[sites, factors])
    # scheduler.add_job(main, 'interval', seconds=100, args=[sites, factors])
   # scheduler.add_job(main, 'cron', hour='08', minute='00', args=[sites, factors], misfire_grace_time=3600)
    scheduler.start()






