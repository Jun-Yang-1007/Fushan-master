#!/usr/bin/env python
# -*- coding:utf-8 -*-
import pandas
import numpy
from sklearn.decomposition import PCA
from keras.models import Sequential
from keras.layers import Dense,BatchNormalization,Activation
from keras.optimizers import Adam,SGD
from keras.layers import LSTM
from keras.metrics import mae,rmse
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import warnings

warnings.filterwarnings('ignore')
import tensorflow as tf

"""
restrict the memory of GPU for avoiding the mistake
"""

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.1
session = tf.Session(config=config)

look_back = 50
look_after = 1

scaler = MinMaxScaler()

def process_data(dataset):
    data = dataset
    dataset = data.astype('float32')
    dataset = numpy.array(dataset)
    dataset = dataset.reshape(-1, 1)
    dataset = scaler.fit_transform(dataset)
    print('dataset.shape after reshape', dataset.shape)
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
    return trainX, trainY


def get_forecast_input(trainx, trainy):
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

    model = Sequential()
    model.add(LSTM(30, input_dim =input_dim))
    model.add(Dense(1))
    model.add(BatchNormalization())
    model.add(Activation('sigmoid'))

    model.compile(loss='rmse', optimizer=Adam(lr=0.0002,beta_1=0.9),
                  metrics=['rmse'])  # (loss="mean_squared_error", optimizer="adam")
    model.fit(trainX, trainY, nb_epoch=epoch,
                        batch_size=64, verbose=1,  shuffle=False)
    return model

def get_next_day(model, input, forecast):
    second_day_input = get_forecast_input(input, forecast)
    second_day_forecast = model.predict(second_day_input)
    second_day_forecast = scaler.inverse_transform(second_day_forecast)
    return {"input": second_day_input, "forecast": second_day_forecast}


def forecast(data):
    data = process_data(data)
    trainx, trainy = get_train_set(data)
    model = train(trainx, trainy, input_dim=look_back, output_dim=look_after, epoch=1)  # epoch=200
    forecast_result = []
    first_day_input = get_forecast_input(trainx, trainy)
    first_day_forecast = model.predict(first_day_input)
    array = []
    array.append({"input": first_day_input, "forecast": first_day_forecast})
    for i in range(4):
        array.append(get_next_day(model, array[-1]['input'], array[-1]['forecast']))

        forecast_result.append(array[i]['forecast'][0][0])
    return forecast_result


def pca(data):
    pca = PCA(n_components = 1)
    paced_data = pca.fit_transform(data)
    print('PCAed training data \n',paced_data[:5])
    inverse_data = pca.inverse_transform(paced_data)
    print('inversed training data \n',inverse_data[:5])
    return paced_data,inverse_data


if __name__ == '__main__':
    dataframe = pandas.read_csv('../newdata.csv')
    data = dataframe.iloc[:,2:7]
    paced_data = pca(data)[0]

    data['pcaed'] = paced_data
    # new_data = pandas.DataFrame(data, data.iloc[0:100,-1].values)
    data['NO.33'] = dataframe.iloc[:,-1]   # 将原来的 NO.33加入到原来的DataFrame里面

    print(data.iloc[:,-2:].head())
    result = forecast(data.iloc[:,-2:])
    print(f'final result -> -> -> -> ->\n{result}')




