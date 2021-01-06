#!/sur/bin/env python
#-*- coding: utf-8 -*-

import datetime
from monthdelta import monthdelta as month

'''
def dateshift(date,delta,format="%Y-%m-%d",freq="D"): # date="YYYY--MM-DD"
    date=datetime.datetime.strptime(date,format)                                       #把时间改成可按日、月、时累加形式
    target  = None
    if freq == 'D':
        target = (date + datetime.timedelta(days = delta)).strftime(format)       #最终时间=预测当天+需要预测的天数
    elif freq == 'M':
        target = (date + monthdelta(delta)).strftime(format)
    elif freq == 'H':
        target = (date + datetime.timedelta(hours = delta)).strftime(format)
    return  target
'''

def dateshift_Hour(date, delta, format="%Y-%m-%d %H:%M:%S"):  # date="YYYY--MM-DD"
    date = datetime.datetime.strptime(date, format)  # 把时间改成可按日、月、时累加形式
    target = None
    target = (date + datetime.timedelta(hours=delta)).strftime(format)
    return target


def dateshift_Day(date, delta, format="%Y-%m-%d"):  # date="YYYY--MM-DD"
    date = datetime.datetime.strptime(date, format)  # 把时间改成可按日、月、时累加形式
    target = None
    target = (date + datetime.timedelta(days=delta)).strftime(format)
    return target


def dateshift_Month(date, delta, format="%Y-%m"):  # date="YYYY--MM-DD"
    date = datetime.datetime.strptime(date, format)  # 把时间改成可按日、月、时累加形式
    target = None

    target = (date + month(delta)).strftime(format)
    return target


