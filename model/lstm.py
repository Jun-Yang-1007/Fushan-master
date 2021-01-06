#!/usr/bin/env python
# -*- coding:utf-8 -*-
import pandas
import numpy
import threading
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import r2_score
from keras.layers.normalization import BatchNormalization
import warnings

warnings.filterwarnings('ignore')
import datetime
from save_db import insert_output

from keras.layers.convolutional import MaxPooling1D, Conv1D
from keras.layers import Flatten, RepeatVector

from keras.metrics import mae   #, rmse
import tensorflow as tf
import os
import GPUtil
import psutil
"""
restrict the memory of GPU for avoiding the mistake
"""
tf.flags.DEFINE_integer('units', '5', 'number of units')
tf.flags.DEFINE_integer('epochs', '2', 'number of epochs')
tf.app.flags.DEFINE_string('activity_log_id', '', 'activity_log_id')
tf.app.flags.DEFINE_string('file_id', '', 'file_id')
FLAGS = tf.flags.FLAGS


config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.1
session = tf.Session(config=config)

look_back = 7
look_after = 1

# scaler = StandardScaler()

scaler = MinMaxScaler(feature_range=(0, 1))


def process_data(dataset):
    data = dataset.ffill().bfill()
    dataset = data.astype('float32')

    dataset = numpy.array(dataset)
    dataset = dataset.reshape(-1, 1)
    print('dataset.shape after reshape', dataset.shape)
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


def get_train_set(dataset, scale=0.8):
    train_size = int(len(dataset) * scale)
    train = dataset[0:train_size, :]
    test = dataset[train_size:, :]
    global testX, testY
    trainX, trainY = create_dateset(train, look_back, look_after)
    testX, testY = create_dateset(test, look_back, look_after)
    trainX = numpy.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
    testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
    print(trainX.shape, trainY.shape)
    return trainX, trainY, testX, testY

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


def train(trainX=None, trainY=None, input_dim=None, output_dim=None, epoch=None):  # epoch=100
    # put all data training model
    keras.backend.clear_session()
    model = Sequential()
    model.add(LSTM(20, input_dim =input_dim))
    model.add(BatchNormalization(momentum=0.99))
    model.add(Dense(1))
    #   print(model.summary())
    model.compile(loss='mse', optimizer='adam',
                  metrics=['mae'])  # (loss="mean_squared_error", optimizer="adam")
    es = EarlyStopping(monitor='val_loss', min_delta=1e-10, patience=10, verbose=1)
    # 如果验证损失在10轮内都没有改善，那么就触发这个回调函数，触发时将学习率除以10
    rlr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, verbose=1)
    global history
    history = model.fit(trainX, trainY, nb_epoch=epoch,
                        batch_size=128, verbose=1, validation_split=0.1, shuffle=False)
    return model


def get_filter_weights(model, layer=None):
    """用于返回 Keras 模型中一个或者所有卷积层的权重的函数"""
    if layer or layer == 0:
        weight_array = model.layers[layer].get_weights()[0]

    else:
        weights = [model.layers[layer_ix].get_weights()[0] for layer_ix in range(len(model.layers)) \
                   if 'conv' in model.layers[layer_ix].name]
        weight_array = [numpy.array(i) for i in weights]
        print('weight_array', weight_array)
    return weight_array


def r2(train_predict, trainx):
    mean = numpy.mean(trainx)
    fenzi = numpy.sum(((train_predict - trainx) ** 2))
    fenmu = numpy.sum(((trainx - numpy.mean(trainx)) ** 2))
    r2 = 1 - (fenzi / fenmu)
    return r2


def forecast(data, type='daily'):
    data = process_data(data)
    trainx, trainy, testX, testY = get_train_set(data)
    # print(f'trainx.shape{trainx.shape}', f'trainy.shape{trainy.shape}')
    # print(trainx[:10],'\n',trainy[:10])
    model = train(trainx, trainy, input_dim=look_back, output_dim=look_after, epoch=1)  # epoch=200
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



    if type == 'daily':
        five_day_input = get_forecast_input(fouth_day_input, fouth_day_forecast)
        five_day_forecast = model.predict(five_day_input)

        six_day_input = get_forecast_input(five_day_input, five_day_forecast)
        six_day_forecast = model.predict(six_day_input)



    first_day_forecast = scaler.inverse_transform(first_day_forecast)
    second_day_forecast = scaler.inverse_transform(second_day_forecast)
    third_day_forecast = scaler.inverse_transform(third_day_forecast)
    fouth_day_forecast = scaler.inverse_transform(fouth_day_forecast)

    if type == 'month':
        five_day_forecast = scaler.inverse_transform(five_day_forecast)
        six_day_forecast = scaler.inverse_transform(six_day_forecast)


    if type == 'daily':
        five_day_forecast = scaler.inverse_transform(five_day_forecast)
        six_day_forecast = scaler.inverse_transform(six_day_forecast)


    forecast_result.append(first_day_forecast)
    forecast_result.append(second_day_forecast)
    forecast_result.append(third_day_forecast)
    forecast_result.append(fouth_day_forecast)

    if type == 'month':
        forecast_result.append(five_day_forecast)
        forecast_result.append(six_day_forecast)


    if type == 'daily':
        forecast_result.append(five_day_forecast)
        forecast_result.append(six_day_forecast)


    forecast_result = numpy.array(forecast_result)
    count = 7

    if type == 'month':
        count = 7

    if type == 'daily':
        count = 7
    #
    result = {'first_day': str(first_day_forecast[0][0]), 'second_day': str(second_day_forecast[0][0]), \
              'third_day': str(third_day_forecast[0][0]), 'fouth_day': str(fouth_day_forecast[0][0]), \
              'fifth_day': str(five_day_forecast[0][0]), 'sixth_day': str(six_day_forecast[0][0])}
    print(f'result{result}')
    forecast_result = numpy.reshape(forecast_result, (6))
    output = {'activity_log_id': FLAGS.activity_log_id, 'output': result}
    insert_output(output)

    # global train_predict, test_predict
    # train_predict = model.predict(trainx)
    # test_predict = model.predict(testX)
    # rms_score = numpy.sqrt(numpy.mean((trainy - train_predict) ** 2))
    # rms_score2 = numpy.sqrt(numpy.mean((testY - test_predict) ** 2))
    # print('training_rmse:', rms_score)
    # print('testing_rmse:', rms_score2)
    # # global r2score
    # # r2score = r2(train_predict, trainy)
    # # print('training_r2score',r2score)
    # # r2score = r2(test_predict, testY)
    # # print('testing_r2score',r2score)
    # print('mape:', 100 * numpy.mean(numpy.abs((testY - test_predict) / numpy.abs(testY), None)))
    # print('training_mae:', numpy.mean(abs((trainy - train_predict))))  # np.mean(abs(test_y1 - predict1))
    # print('testing_mae:', numpy.mean(abs((testY - test_predict))))
    # print("The R2 score on the Train set is:\t{:0.3f}".format(r2_score(trainy, train_predict)))
    # print('************-------------***************', trainy.shape, train_predict.shape)
    # print('************-------------***************', testY.shape, test_predict.shape)
    # test_r2_score = r2_score(testY, test_predict)
    # print("The R2 score on the Test set is:\t{:0.3f}".format(test_r2_score))
    #
    # # timer = threading.Timer(86400, forecast(data))
    # # timer.start()
    # print('前五个预测值为', forecast_result[:5])
    #
    # # huatu
    # import matplotlib.pyplot as plt
    # plt.figure(dpi=300)
    # plt.plot(train_predict, label='train_predict_data', linewidth='0.6', c='red')  # 将不同结果放在一起，一个真实值，两种预测值
    # plt.plot(trainy, label='train_raw_data', linewidth='0.6', c='green')
    # plt.legend()
    # plt.show()
    #
    # test_predict = scaler.inverse_transform(test_predict)
    # testy = scaler.inverse_transform(testY)
    # plt.plot(test_predict, label='test_predict_data', linewidth='0.3', c='red')  # 将不同结果放在一起，一个真实值，两种预测值
    # plt.plot(testy, label='test_raw_data', linewidth='0.3', c='green')
    # plt.legend()
    # plt.show()
    #
    # plt.plot(history.history['loss'], label='train')
    # plt.plot(history.history['val_loss'], label='test')
    # plt.legend(['train', 'test'])
    # plt.legend()
    # plt.show()
    #
    return forecast_result



def get_gpu_info():
    '''
    :return:
    '''
    Gpus = GPUtil.getGPUs()
    gpulist = []
    GPUtil.showUtilization()
    for gpu in Gpus:
        # print('gpu.id:', gpu.id)
        print('GPU总量：', gpu.memoryTotal*1024*1024) # 转化为byte
        print('GPU使用量：', gpu.memoryUsed*1024*1024)
        print('gpu使用占比:', gpu.memoryUtil * 100)
        gpulist.append([gpu.memoryTotal, gpu.memoryUsed, gpu.memoryUtil * 100])
    return gpulist


def get_cpu_info():
    mem_available = psutil.virtual_memory().total
    mem_process = psutil.Process(os.getpid()).memory_info().rss
    cpu_res = psutil.cpu_percent()
    print('CPU总量：', mem_available)
    print('CPU使用量：', mem_process)
    print('CPU使用占比:', round(mem_process/mem_available,3) * 100)
    print('CPU使用率：', cpu_res)
    return round(mem_process / 1024 / 1024, 2), round(mem_available / 1024 / 1024, 2)


def cpu_gpu_monitor():
    gpu_info = get_gpu_info()
    cpu_info = get_cpu_info()
    print(gpu_info, '\n')
    print(cpu_info, '\n')


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
'''

'''
通过web启动，通过web多次请求即可成为定时任务
'''
# from fastapi import FastAPI
# app = FastAPI()
# @app.get("/")     # 修饰名称(http://127.0.0.1:8000/12345)
# async def root():
#     dataframe = pandas.read_csv('C:/Users/41634/Desktop/fushandata/hour.csv')
#     data = dataframe['F_TP']
#     result = forecast(data, type='daily').tolist()
#     return {"message": result}

# if __name__ == '__main__':
#     uvicorn.run(app='lstm:app', host="0.0.0.0", port=8000, reload=True)


if __name__ == '__main__':
    dataframe = pandas.read_csv('../newdata.csv')
    data = dataframe.iloc[:,2:8]
    print(f'data.shape{data.shape}')
    result = forecast(data, type='daily')
    #   weight_array = get_filter_weights(model, layer='conv1d_1')
    print(result)

    cpu_gpu_monitor()

# # 在 2019-8-30 01:00:01 运行一次 job 方法
# scheduler = BlockingScheduler()
# scheduler.add_job(forecast, 'interval', seconds=10, args=[data])
# # scheduler.add_job(forecast, 'cron', hour="12", args=[data])
# scheduler.start()

