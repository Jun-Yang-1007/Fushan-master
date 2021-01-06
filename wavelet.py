import pywt
import pandas as pd
import talib as ta
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import  warnings

import time
warnings.filterwarnings('ignore')
import datetime
start = time.time()

# fig = plt.figure(dpi=300)      #设置图形大小
# plt.plot(range(len(df)), df, label='NO.13',c= 'darkorange')
# plt.legend()
# plt.show()


def threshold_cal(signal):
    # rigrsure方法计算阈值
    signal = abs(signal)
    signal.sort()
    signal = signal**2
    list_risk_j = []; N = len(signal)
    for j in range(N):
        if j == 0:
            risk_j = 1 + signal[N-1]
        else:
            risk_j = (N-2*j + (N-j)*(signal[N-j]) + sum(signal[:j]))/N
        list_risk_j.append(risk_j)
    k = np.array(list_risk_j).argmin()
    threshold = np.sqrt(signal[k])
    return threshold


def wavelet_cal(data):
    global res_4
    res_4 = pywt.wavedec(np.array(data),wavelet='sym8',mode='symmetric',level=4)
    for j in [3, 4]:  # 对于高频的信号进行阈值处理
        signal = np.array(res_4[j])
        threshold = threshold_cal(signal)  # 固定阈值方法 np.sqrt(2*np.log(len(signal)))
        res_4[j] = pywt.threshold(signal, threshold, 'soft')
    rec_szzz = pywt.waverec(res_4, 'sym8')
    return rec_szzz

def display(data):
    for j in [1,2,3,4]:
        plt.subplot(2,2,j)
        plt.plot(range(len(data[j])),data[j],c='lightgreen')
        plt.title('part'+str(j))
    plt.show()

if __name__ == '__main__':
    df = pd.read_csv('C:/Users/41634/Desktop/paper/newdata.csv')
    df['date'] = pd.to_datetime(df['date'])
    df = df['NO.13']
    print('111')
    data = wavelet_cal(df)
    print(data)
    plt.plot(range(len(data)), data, c='lightgreen')
    plt.show()


'''

df = rec_szzz
# df=pd.read_csv('C:/Users/41634/Desktop/paper/newdata.csv')
# df['date']=pd.to_datetime(df['date'])
# df=df['NO.8']
data_all = np.array(df).astype(float)

data=[]
for i in range(len(data_all)- 10 -1):
    data.append(data_all[i:i+ 10])
reshaped_data=np.array(data).astype(float)
MMS=MinMaxScaler()


x=reshaped_data[:,:-1]
print(x.shape)
y=reshaped_data[:,-1]
y=y.reshape(-1,1)
print(y.shape)
x=MMS.fit_transform(x)
y=MMS.fit_transform(y)

split_boundary=int(reshaped_data.shape[0]*0.8)
train_x=x[:split_boundary]
test_x=x[split_boundary:]
train_y=y[:split_boundary]
test_y=y[split_boundary:]

train_x=np.reshape(train_x,(train_x.shape[0], train_x.shape[1],1))
test_x=np.reshape(test_x,(test_x.shape[0],test_x.shape[1],1))


with tf.name_scope("LSTM_layers"):

            LSTM_layers = [50]
            Conv1D_layers = [64]
            i = 1
         #   if i < len(LSTM_layers)* len(Conv1D_layers)+1:
            for LSTM_layer in LSTM_layers:
                for Conv1D_layer in Conv1D_layers:
                    model_name = "LSTM-{}-nodes-{}-dense-{}".format(LSTM_layer,Conv1D_layer,datetime.datetime.now().strftime(datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")))  #
                    global tensorboard
                    tensorboard = TensorBoard(log_dir='logs/{}'.format(model_name))
                    print(model_name)
                    model = Sequential()
                    model.add(Conv1D(filters=Conv1D_layer, kernel_size=1, activation='linear', input_shape=(9,1)))
                    model.add(MaxPooling1D(pool_size=1))
                    model.add(Flatten())
                    model.add(RepeatVector(1))
                    # model.add(LSTM(25, activation='linear', return_sequences=True))
                    model.add(LSTM(LSTM_layer, activation='linear', return_sequences=True))
                    model.add(Flatten())
                    model.add(Dense(1))
                    model.compile(loss='rmse', optimizer='adam', metrics=['mape','rmse'])
                    i+=1


                    model.fit(train_x,train_y,batch_size=64, epochs=3,validation_split=0.2) #,callbacks=[tensorboard]   32
                    predict2=model.predict(test_x)
                    predict2=MMS.inverse_transform(predict2)
                    test_y2=MMS.inverse_transform(test_y)
                    plt.figure(dpi=500)
                  #  plt.plot(predict1, label='predict_data1', linewidth='0.4', c='green')   #将不同结果放在一起，一个真实值，两种预测值
                  #   plt.plot(predict2, label='predict-data', linewidth='0.6', c = 'blue')
                  #   plt.plot(test_y2, label='raw_data', linewidth='0.6', c = 'darkorange')
                  #   plt.legend()
                  #   plt.show()

             #       print(r2_score(test_y1, predict1))
                    print( 'r12:',r2_score(test_y2, predict2))

                    print('rmse:', np.sqrt(np.mean((test_y2 -  predict2) ** 2)))
                    print('mape:', 100 * np.mean(np.abs((test_y2 -  predict2) / np.abs(test_y2))))
                    print('mae:', np.mean(np.abs((test_y2 - predict2))))

end = time.time()
print(f'消耗时间{end-start}')

def change(number):
    A  = []
    for num in number:
        A.append(num[0])
    return A

predict22 = change(predict2)
test_y22 = change(test_y2)


data2 = {"predict-data": predict22, 'raw_data': test_y22}
data2 = pd.DataFrame(data2)
plt.figure(dpi=300)
sns.lmplot(y='raw_data', x='predict-data', data=data2, line_kws={'color':'#01a2d9','alpha':0.2},scatter_kws={ 's':1, 'color':'darkorange', 'alpha':0.7},ci=95,markers=['+'])
#sns.stripplot(y='raw_data', x='predict_data', data=data)
plt.show()

'''


