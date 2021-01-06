#!/usr/bin/env python
# -*- coding:utf-8 -*-
import pandas
import numpy
from sklearn.decomposition import PCA
from keras.models import Sequential
from keras.layers import Dense,BatchNormalization,Activation
from keras.optimizers import Adam,SGD
from keras.layers import LSTM
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

look_back = 7
look_after = 1



def process_data(dataset):
    data = dataset
    dataset = data.astype('float32')
    dataset = numpy.array(dataset)
    dataset = dataset.reshape(-1, 1)
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
    model.add(LSTM(20, input_dim =input_dim))
    model.add(Dense(1))
    model.add(BatchNormalization())
    model.add(Activation('sigmoid'))

    model.compile(loss='mse', optimizer=Adam(lr=0.002,beta_1=0.9),
                  metrics=['mae'])  # (loss="mean_squared_error", optimizer="adam")
    model.fit(trainX, trainY, nb_epoch=epoch,
                        batch_size=128, verbose=1,  shuffle=False)
    return model

def get_next_day(model, input, forecast):
    second_day_input = get_forecast_input(input, forecast)
    second_day_forecast = model.predict(second_day_input)
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
    for i in range(6):
        array.append(get_next_day(model, array[-1]['input'], array[-1]['forecast']))

        forecast_result.append(array[i]['forecast'][0][0])
    return forecast_result


# def forecast(data, type='daily'):
#     data = process_data(data)
#     trainx, trainy, testX, testY = get_train_set(data)
#
#     model = train(trainx, trainy, input_dim=look_back, output_dim=look_after, epoch=1)  # epoch=200
#     forecast_result = []
#     first_day_input = get_forecast_input(trainx, trainy)
#     first_day_forecast = model.predict(first_day_input)
#     forecast_result.append(first_day_forecast)
#     forecast_result = numpy.reshape(forecast_result, (1))
#     return forecast_result


def pca(data):
    pca = PCA(n_components = 1)

    paced_data = pca.fit_transform(data)
    print('PCAed training data \n',paced_data[:5])

    inverse_data = pca.inverse_transform(paced_data)
    print('inversed training data \n',inverse_data[:5])

    return paced_data,inverse_data


if __name__ == '__main__':
    dataframe = pandas.read_csv('../newdata.csv')
    data = dataframe.iloc[0:100,2:8]
    pca(data)


    # PCA_data = pca(data)[0]
    # result = forecast(PCA_data)
    # inversed_data = pca([result])[1]
    # print(f'final result -> -> -> -> ->{result}')
    # print(f'inversed data =========》{inversed_data}')
    # print(f'********》{pca(PCA_data)[1]}')



    # new_data = pandas.DataFrame(PCA_data,data.iloc[0:,-1].values)
    # # 把要预测的特征和混合特征塞在一起，存起来再读取。直接读的时候变成只是一列（实际两列）
    # new_data.iloc[:,0].to_csv('./PCAdata.csv')
    # combination_data = pandas.read_csv('./PCAdata.csv',index_col=0)
    # result = forecast(combination_data)
    # print(result)
    # print(pca(data.iloc[:,:-1])[1])  # 反变化
   # print(pca(numpy.reshape(result,(-1,1)))[1])  # 反变化 不然会报reshape错误。 或者改成[result]  不用 reshape

