#!/usr/bin/env python
# -*- coding:utf-8 -*-

import pandas
import numpy
import threading
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from apscheduler.schedulers.blocking import BlockingScheduler
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import r2_score
from keras.metrics import mape, rmse, mae
import warnings

warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt

# import os
# os.environ['CUDA_VISIBIE_DEVICES']="1,2"
# look_back >= look_after
look_back = 7
look_after = 1

scaler = StandardScaler()


# scaler = MinMaxScaler(feature_range=(0, 1))

def process_data(dataset):
    # data = dataset.resample('D').fillna(method='ffill')
    # data = dataset.resample('H').fillna(method='ffill')

    data = dataset.fillna(method='ffill')
    dataset = data.astype('float32')
    # dataset = numpy.reshape(dataset, (dataset.shape[0], 1))        ###把数据转成 n行1列
    dataset = numpy.array(dataset)
    dataset = dataset.reshape(-1, 1)
    dataset = scaler.fit_transform(dataset)
    return dataset


# create data set which used for trainning and testing.
def create_dateset(dataset, look_back=7, look_after=1):  ###train_split_traing test
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back):
        x = dataset[i:(i + look_back), 0]
        y = dataset[(i + look_back):(i + look_after + look_back), 0]
        dataX.append(x)
        dataY.append(y)
    return numpy.array(dataX), numpy.array(dataY)


def get_train_set(dataset, scale=0.9):
    train_size = int(len(dataset) * scale)
    train = dataset[0:train_size, :]
    test = dataset[train_size:, :]
    trainX, trainY = create_dateset(train, look_back, look_after)
    global testX, testY
    testX, testY = create_dateset(test, look_back, look_after)
    trainX = numpy.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))  ## ？？
    testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
    print(trainX.shape, trainY.shape)
    return trainX, trainY, testX, testY


def train(trainX, trainY, input_dim=1, output_dim=1, epoch=10):  # epoch=100
    # put all data training model
    keras.backend.clear_session()
    model = Sequential()
    model.add(LSTM(50, input_dim=input_dim, return_sequences=True))
    model.add(LSTM(10))
    model.add(Dense(output_dim))
    model.compile(loss='mse', optimizer='adam')  # (loss="mean_squared_error", optimizer="adam")
    # model.fit(trainX, trainY, nb_epoch=epoch, batch_size=128, verbose=1,validation_split=0.1)
    history = model.fit(trainX, trainY, nb_epoch=epoch, batch_size=1, verbose=1, validation_data=(testX, testY))

    return model, history


def get_forecast_input(trainx, trainy):
    # trainx [[[1,2,3]],[[4,5,6]]]
    # trainy [[1,2,3],[4,5,6]]
    trainx = trainx[trainx.shape[0] - 1][trainx.shape[1] - 1]
    trainy = trainy[trainy.shape[0] - 1]

    input = []

    for i in range(look_after, look_back):
        input.append(trainx[i])
    for i in range(look_after):
        input.append(trainy[i])
    input = numpy.reshape(input, (1, 1, look_back))

    return input


def r2(y_prd, y):
    Residual = numpy.sum((y - y_prd) ** 2)  # 残差平方和
    total = numpy.sum((y - numpy.mean(y)) ** 2)  # 总体平方和
    R_square = 1 - (Residual / total)  # 相关性系数R^2
    return R_square


def forecast(data, type=None):
    data = process_data(data)
    trainx, trainy, testx, testy = get_train_set(data)
    model, history = train(trainx, trainy, input_dim=look_back, output_dim=look_after, epoch=50)  # epoch=200
    # train(trainX, trainY, input_dim=1, output_dim=1, epoch=100):
    train_predict = model.predict(trainx)
    test_predict = model.predict(testx)

    train_predict = scaler.inverse_transform(train_predict)
    trainy = scaler.inverse_transform(trainy)
    rms_score = numpy.sqrt(numpy.mean((trainx - train_predict) ** 2))
    print('rmse:', rms_score)
    r2score = r2(train_predict, trainy)
    print('r2score', r2score)
    print('mape:', 100 * numpy.mean(numpy.abs((trainx - train_predict) / numpy.abs(trainx), None)))

    plt.plot(train_predict, label='train_predict_data', linewidth='1', c='green')  # 将不同结果放在一起，一个真实值，两种预测值
    plt.plot(trainy, label='train_raw_data', linewidth='0.4', c='darkorange')
    plt.legend()
    plt.show()

    test_predict = scaler.inverse_transform(test_predict)
    testy = scaler.inverse_transform(testy)
    plt.plot(test_predict, label='test_predict_data', linewidth='1', c='green')  # 将不同结果放在一起，一个真实值，两种预测值
    plt.plot(testy, label='test_raw_data', linewidth='0.4', c='darkorange')
    plt.legend()
    plt.show()

    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='test')
    plt.legend(['train', 'test'])
    plt.legend()#!/usr/bin/env python
# -*- coding:utf-8 -*-

import pandas
import numpy
import threading
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from apscheduler.schedulers.blocking import BlockingScheduler
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings('ignore')
from database.missing_value_filling import set_missing
import datetime

import os
# os.environ['CUDA_VISIBIE_DEVICES']="1,2"
# look_back >= look_after

from keras.layers.convolutional import MaxPooling1D, Conv1D
from keras.layers import  Flatten,RepeatVector

from keras.metrics import mae, rmse

look_back = 7
look_after = 1

scaler = StandardScaler()

#scaler = MinMaxScaler(feature_range=(0, 1))



def process_data(dataset,type = 'hourly'):
    if type == 'daily':
        data = dataset.resample('D').ffill()
    if type == 'hourly':
        data = dataset.resample('H').ffill()
    dataset = data.astype('float32')
    dataset = numpy.array(dataset)
    dataset = dataset.reshape(-1, 1)
    print('dataset.shape after reshape', dataset.shape)
    dataset = scaler.fit_transform(dataset)
    return dataset



# create data set which used for trainning and testing.
def create_dateset(dataset, look_back=1, look_after=1):                       ###train_split_traing test
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back):
        x = dataset[i:(i+look_back), 0]
        y = dataset[(i+look_back):(i+ look_after+look_back), 0]
        dataX.append(x)
        dataY.append(y)
    return numpy.array(dataX), numpy.array(dataY)



def get_train_set(dataset, scale=0.8):
    train_size = int(len(dataset) * scale)
    train = dataset[0:train_size, :]
    test = dataset[train_size:, :]
    global testX, testY
    trainX, trainY = create_dateset(train, look_back, look_after)
    testX, testY = create_dateset(test, look_back, look_after)
    trainX = numpy.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
    testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
    return trainX, trainY,testX, testY


def train(trainX, trainY,input_dim=1, output_dim=1, epoch=100):   #epoch=100
    # put all data training model
    keras.backend.clear_session()
    model = Sequential()

    # model.add(Conv1D(filters=16, kernel_size=1, activation='linear', input_shape=(1,7)))
    # model.add(MaxPooling1D(pool_size=1))
    # model.add(Flatten())
    # model.add(RepeatVector(1))  # 将输入重复一次

    model.add(LSTM(20, input_dim=input_dim))
    model.add(Dense(output_dim))
    model.compile(loss='mse', optimizer='adam', metrics=['mae','rmse'])                #(loss="mean_squared_error", optimizer="adam")

    global history
    history = model.fit(trainX, trainY, nb_epoch=epoch, batch_size=1, verbose=1, validation_split=0.1,shuffle=False)
    return model


def get_forecast_input(trainx, trainy):
    # trainx [[[1,2,3]],[[4,5,6]]]
    # trainy [[1,2,3],[4,5,6]]
    trainx = trainx[trainx.shape[0] - 1][trainx.shape[1] - 1]
    trainy = trainy[trainy.shape[0] - 1]
    input = []

    for i in range(look_after, look_back):
        input.append(trainx[i])
    for i in range(look_after):
        input.append(trainy[i])
    input = numpy.reshape(input, (1, 1, look_back))
    return input


def r2(train_predict,trainx):
    mean = numpy.mean(trainx)
    fenzi = numpy.sum(((train_predict - trainx)**2))
    fenmu = numpy.sum(((trainx - numpy.mean(trainx))**2))
    r2 = 1 - (fenzi /fenmu)
    return r2



def forecast(data, type='daily'):
    data = process_data(data,type = 'hourly')
    trainx, trainy, testX, testY = get_train_set(data)
    model = train(trainx, trainy, input_dim=look_back, output_dim=look_after, epoch=5)    #epoch=200
            #train(trainX, trainY, input_dim=1, output_dim=1, epoch=100):


    # i = 1
    # while i<3:
    #     if test_r2_score<0.5:
    #         print('need re-train')
    #         model = train(trainx, trainy, input_dim=look_back, output_dim=look_after, epoch=20)  # epoch=200
    #         # train(trainX, trainY, input_dim=1, output_dim=1, epoch=100):
    #
    #         train_predict = model.predict(trainx)
    #         test_predict = model.predict(testX)
    #         rms_score = numpy.sqrt(numpy.mean((trainy - train_predict) ** 2))
    #         print('rmse:', rms_score)
    #
    #         r2score = r2(train_predict, trainy)
    #         print('r2score', r2score)
    #         print('mape:', 100 * numpy.mean(numpy.abs((trainy - train_predict) / numpy.abs(trainy), None)))
    #         print("The R2 score on the Train set is:\t{:0.3f}".format(r2_score(trainy, train_predict)))
    #         print("The R2 score on the Test set is:\t{:0.3f}".format(r2_score(testY, test_predict)))
    #     else:
    #         print('well done the model')
    #     i +=1

    forecast_result = []

    first_day_input = get_forecast_input(trainx, trainy)
    first_day_forecast = model.predict(first_day_input)

    second_day_input = get_forecast_input(first_day_input, first_day_forecast)
    second_day_forecast = model.predict(second_day_input)

    third_day_input = get_forecast_input(second_day_input, second_day_forecast)
    third_day_forecast = model.predict(third_day_input)

    fouth_day_input = get_forecast_input(third_day_input, third_day_forecast)
    fouth_day_forecast = model.predict(fouth_day_input)

    if type == 'month':
        five_day_input = get_forecast_input(fouth_day_input, fouth_day_forecast)
        five_day_forecast = model.predict(five_day_input)

        six_day_input = get_forecast_input(five_day_input, five_day_forecast)
        six_day_forecast = model.predict(six_day_input)

        seven_day_input = get_forecast_input(six_day_input, six_day_forecast)
        seven_day_forecast = model.predict(seven_day_input)

    if type == 'daily':
        five_day_input = get_forecast_input(fouth_day_input, fouth_day_forecast)
        five_day_forecast = model.predict(five_day_input)

        six_day_input = get_forecast_input(five_day_input, five_day_forecast)
        six_day_forecast = model.predict(six_day_input)

        seven_day_input = get_forecast_input(six_day_input, six_day_forecast)
        seven_day_forecast = model.predict(seven_day_input)



    first_day_forecast = scaler.inverse_transform(first_day_forecast)
    second_day_forecast = scaler.inverse_transform(second_day_forecast)
    third_day_forecast = scaler.inverse_transform(third_day_forecast)
    fouth_day_forecast = scaler.inverse_transform(fouth_day_forecast)

    if type == 'month':
        five_day_forecast = scaler.inverse_transform(five_day_forecast)
        six_day_forecast = scaler.inverse_transform(six_day_forecast)
        seven_day_forecast = scaler.inverse_transform(seven_day_forecast)

    if type == 'daily':
        five_day_forecast = scaler.inverse_transform(five_day_forecast)
        six_day_forecast = scaler.inverse_transform(six_day_forecast)
        seven_day_forecast = scaler.inverse_transform(seven_day_forecast)

    forecast_result.append(first_day_forecast)
    forecast_result.append(second_day_forecast)
    forecast_result.append(third_day_forecast)
    forecast_result.append(fouth_day_forecast)

    if type == 'month':
        forecast_result.append(five_day_forecast)
        forecast_result.append(six_day_forecast)
        forecast_result.append(seven_day_forecast)

    if type == 'daily':
        forecast_result.append(five_day_forecast)
        forecast_result.append(six_day_forecast)
        forecast_result.append(seven_day_forecast)

    forecast_result = numpy.array(forecast_result)
    count = 7

    if type == 'month':
        count = 7

    if type == 'daily':
        count = 7

    forecast_result = numpy.reshape(forecast_result, (count))


    global train_predict,test_predict
    train_predict = model.predict(trainx)
    test_predict = model.predict(testX)
    rms_score= numpy.sqrt(numpy.mean((trainy - train_predict )**2))
    rms_score2 = numpy.sqrt(numpy.mean((testY - test_predict) ** 2))
    print('training_rmse:',rms_score)
    print('testing_rmse:', rms_score2)
    # global r2score
    # r2score = r2(train_predict, trainy)
    # print('training_r2score',r2score)
    # r2score = r2(test_predict, testY)
    # print('testing_r2score',r2score)
    print('mape:', 100 * numpy.mean(numpy.abs((testY- test_predict) / numpy.abs(testY), None)))
    print('training_mae:', numpy.mean(abs((trainy - train_predict))))  # np.mean(abs(test_y1 - predict1))
    print('testing_mae:', numpy.mean(abs((testY - test_predict))))
    print("The R2 score on the Train set is:\t{:0.3f}".format(r2_score(trainy, train_predict)))
    print('************-------------***************', trainy.shape, train_predict.shape)
    print('************-------------***************',testY.shape,test_predict.shape )
    test_r2_score = r2_score(testY,test_predict)
    print("The R2 score on the Test set is:\t{:0.3f}".format(test_r2_score))

    # timer = threading.Timer(86400, forecast(data))
    # timer.start()
    print('前五个预测值为',forecast_result[:5])
    return forecast_result


# # create data set which used for trainning and testing.
# def create_dateset(dataset, look_back=1, look_after=1):                       ###train_split_traing test
#     dataX, dataY = [], []
#     for i in range(len(dataset) - look_back):
#         x = dataset[i:i + look_back, 0]
#         y = dataset[i + look_back:i + look_after + look_back, 0]
#         dataX.append(x)
#         dataY.append(y)
#     return numpy.array(dataX), numpy.array(dataY)

'''
    import os, time, logging
    from  database import  connection
    output_dir = './'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    log_name = '{}.log'.format(time.strftime('%Y-%m-%d-%H-%M'))
    final_log_file = os.path.join(output_dir,log_name)
    # creat a log
    log = logging.getLogger('train_log')
    log.setLevel(logging.DEBUG)
     # FileHandler
    file = logging.FileHandler(final_log_file)
    file.setLevel(logging.DEBUG)
     # StreamHandler
    stream = logging.StreamHandler()
    stream.setLevel(logging.DEBUG)
     # Formatter
    formatter = logging.Formatter(
         '[%(asctime)s][line: %(lineno)d] ==> %(message)s')
     # setFormatter
    file.setFormatter(formatter)
    stream.setFormatter(formatter)
     # addHandler
    log.addHandler(file)
    log.addHandler(stream)
    # log.info('predici'.format(final_log_file))
    log.info('prediction data {}'.format(history))
    print('/////////////////////////////////////////')
'''


##日志输出
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
    fileName = datetime.datetime.now().strftime('day' + '%Y_%m_%d')
    sys.stdout = Logger(fileName + '.lstmlog', path=path)



if __name__ == '__main__':
    section = "1"
    factor ='DO'

    # fix random seed for reproducibility
    numpy.random.seed(0)

    dateparse = lambda dates: pandas.datetime.strptime(dates, '%m/%d/%Y')
    # # plot data set
    # dataframe = pandas.read_csv('E:/MyFpi/Project1/fushan/Data/section_' + section + '_day_data.csv',
    #                             parse_dates=['date'], index_col='date', date_parser=dateparse)
    dataframe = pandas.read_csv('../section_1_day_data.csv',
                                parse_dates=['date'], index_col='date', date_parser=dateparse)
    data = dataframe['DO'][0:100]
    # data = data['2015-01-01':'2015-02-28'].interpolate()

    # make_print_to_file(path='./')  #输出日志
    result=forecast(data,type='daily')
    print(result)



    # # 在 2019-8-30 01:00:01 运行一次 job 方法
    # scheduler = BlockingScheduler()
    # scheduler.add_job(forecast, 'interval', seconds=10, args=[data])
    # # scheduler.add_job(forecast, 'cron', hour="12", args=[data])
    # scheduler.start()



