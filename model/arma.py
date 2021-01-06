#!/sur/bin/env python
#-*- coding: utf-8 -*-


'''
1. 读取数据，重采样填补空值，标准化数据  def process_data(dataset,freq='D')
2.翻转标准化，用于预测后加入原数据   def reconver_data
3.计算ARMA的阶数，获得AIC(I,J)    def calculate_order(dataset)
4.通过阶数建模                     def predict
5.预测，预测后反标准化           def forecast(dataset,delta=2,freq='D')

用于做训练集的数据集是预测当天的前一个时间段，
'''



import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.arima_model import ARMA
from sklearn.preprocessing import MinMaxScaler,StandardScaler
import os,sys
expand = 1000.0

scaler = StandardScaler()


def process_data(dataset,freq='H'):                       #归一化   dataset=log(A*1000)
    dataset =dataset.resample(freq).ffill()                   #按天重采样，可能有缺失的天数，用fill()填充
    dataset = dataset * expand
    dataset = np.log(dataset)
    return dataset
                      
def reconver_data(dataset):                                  #反归一化   EXP(dataset)/1000
    log_recover = np.exp(dataset)
    dataset = log_recover / expand
    return dataset



def calculate_order(dataset):
    try:    
        res = sm.tsa.arma_order_select_ic(dataset,ic=['aic','bic'],trend='nc')
        aic = res.aic_min_order
        bic = res.bic_min_order
        print("---------------------")
        print("AIC:{0},BIC:{1}".format(aic,bic))
        return aic
    except IndexError as msg:
        print(msg)
        print("Auto calculate P,Q Failed.")
        return None


def predict(dataset,days=7):   #days=2
    order = calculate_order(dataset)
    if order is None:
        print("Failed forecast, retuen None.")
        return None
    
    flag = True
    success = False
    
    p = order[0]
    q = order[1]

    for i in range(p,-1,-1):
        if success == True:
            break
        for j in range(q,-1,-1):    
            if success == True:
                break
            try:

                order = (i,j)

                print("ARMA Try ORDER:{0}".format(order))
                model = ARMA(dataset, order=order)               #  model = ARMA(data,   (1,0))     order=(i,j) 有两个参数
                result_arma = model.fit(disp=-1,method='css')
                success = True

            except  ValueError as msg:
                print("Error :{0},Order:{1}".format(msg,order))

            except :
                print("Unknown Error!")

    if success == True :
        dataset = result_arma.predict(len(dataset), len(dataset) + days, dynamic=True)  # original
        # dataset = result_arma.predict(len(dataset)-days, len(dataset), dynamic=True)  # revised
   
    #源码： def predict(self, start=None, end=None, exog=None, typ='linear',
    #             dynamic=False):
    #     return self.model.predict(self.params, start, end, exog, typ, dynamic)
        print('dataset***********************************',dataset)
        return dataset
    else:
        return None



def forecast(dataset,delta=7,freq='H'):   # delta=5

    dataset = process_data(dataset,freq)
    dataset = predict(dataset,delta)          #def predict(dataset,days=2)   delta为days，可改
    if dataset is None:
        return None

    dataset = reconver_data(dataset)
    print('预测值***********************************',dataset)
    return dataset


if __name__=='__main__':
    dateparse = lambda dates: pd.datetime.strptime(dates, '%m/%d/%Y')

    section = "1"
    factor = 'DO'
    parent_path = os.path.dirname(sys.path[0])
    #parent_path = sys.path[0]
    if parent_path not in sys.path:
        sys.path.append(parent_path)
    print(parent_path)

    data = pd.read_csv('E:/MyFpi/Project1/fujiankehuduan/Data/section_' + section + '_day_data.csv', parse_dates=['date'], index_col='date',
                       date_parser=dateparse)

    tp = data[factor]

    tp_2015 = tp['2015-01-01':'2015-12-28'].interpolate()        #### 由A的数据预测未来天数的情况， 可以选择A的时间段

    #print(tp['2015-01-01':'2015-12-28']) 

    #tp_2015 = tp['2015-11-01':'2015-12-31'].interpolate()
#    print('tp_2015**********************',tp_2015)
    
    result=forecast(tp_2015)

    print ('结果是',result)


