#!/usr/bin/env python
# -*- coding:utf-8 -*-

import pandas
import numpy
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

from sklearn.preprocessing import StandardScaler

import warnings

warnings.filterwarnings('ignore')
import tensorflow as tf

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.3
session = tf.Session(config=config)

look_back = 7
look_after = 1

scaler = StandardScaler()


def process_data(dataset):
    data = dataset.fillna(method='ffill')
    dataset = data.astype('float32')
    dataset = dataset.values
    # dataset = numpy.reshape(dataset, (dataset.shape[0], 1))        ###把数据转成 n行1列
#    dataset = numpy.array(dataset)
    return dataset


# create data set which used for trainning and testing.
def create_dateset(dataset, look_back=1, look_after=1):  ###train_split_traing test
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back):
        x = dataset[i:(i + look_back), 0:dataset.shape[1]]
        y = dataset[(i + look_back):(i + look_after + look_back), 0]
        dataX.append(x)
        dataY.append(y)
    return numpy.array(dataX), numpy.array(dataY)

    # for i in range(n_past, len(data.values) - n_future + 1):
    #     X_train.append(data.values[i - n_past:i, 0:data.shape[1]])
    #     y_train.append(data.values[i + n_future - 1:i + n_future, 0])
    # X_train, y_train = numpy.array(X_train), numpy.array(y_train)


def get_train_set(dataset, scale=1):
    train_size = int(len(dataset) * scale)
    # train = dataset[0:train_size, :]
    train = dataset
    trainX, trainY = create_dateset(train, look_back, look_after)
#    trainX = numpy.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))  ## ？？
    print(trainX.shape, trainY.shape)
    return trainX, trainY


def train(trainX, trainY, input_shape = (7,8), epoch=100):  # epoch=100
    # put all data training model
    keras.backend.clear_session()
    model = Sequential()
    model.add(LSTM(units=10, input_shape = (7,8)))
    model.add(Dense(units=1, activation='linear'))
    model.compile(optimizer='Adam', loss='mean_squared_error')
    model.fit(trainX, trainY, nb_epoch=epoch, batch_size=20, verbose=1, validation_split=0.1)
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
   # input = numpy.reshape(input, (1, 1, look_back))
    return input


def forecast(data):
    data = process_data(data)
    trainx, trainy = get_train_set(data)
    model = train(trainx, trainy, input_shape = (7,8), epoch=5)  # epoch=200
    forecast_result = []

    first_day_input = get_forecast_input(trainx, trainy)
    first_day_forecast = model.predict(first_day_input)
    forecast_result.append(first_day_forecast)

    return forecast_result


if __name__ == '__main__':
    section = "1"
    factor = 'DO'

    dateparse = lambda dates: pandas.datetime.strptime(dates, '%m/%d/%Y')

    dataframe = pandas.read_csv('E:/MyFpi/Project1/fushan/Data/section_' + section + '_day_data.csv',
                                parse_dates=['date'], index_col='date', date_parser=dateparse)
    data = dataframe
    data = data[:100]

    # X_train = []
    # y_train = []

    # n_future = 1  # Number of days we want top predict into the future
    # n_past = 7  # Number of past days we want to use to predict the future
    #
    # for i in range(n_past, len(data.values) - n_future + 1):
    #     X_train.append(data.values[i - n_past:i, 0:data.shape[1]])
    #     y_train.append(data.values[i + n_future - 1:i + n_future, 0])
    # X_train, y_train = numpy.array(X_train), numpy.array(y_train)
    # print('222222222222 \n', X_train[0:1], '3333333333333 \n', y_train)
    # print('X_train shape == {}.'.format(X_train.shape))
    # print('y_train shape == {}.'.format(y_train.shape))
    # print(X_train[0:1]==dataX[0:1])




    result = forecast(data)
    print(result)
