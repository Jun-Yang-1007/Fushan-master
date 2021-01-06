import  pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller as ADF
from statsmodels.stats.diagnostic import acorr_ljungbox

from statsmodels.tsa.arima_model import ARMA
from statsmodels.tsa.stattools import arma_order_select_ic
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')

 
file=pd.read_csv(r'E:\MyFpi\Project1\fujian-water-master-PY3\Data\section_1_day_data.csv')
file=pd.read_csv('C:/Users/41634/Desktop/paper/newdata.csv')
file=file.dropna()
data=file.iloc[:90]#取到12月21号的数据，留下10天验证
data.day=pd.to_datetime(data.date)


#画时序图
plt.plot(data.date,data['NO.8'])
plt.xlabel('day')
plt.ylabel('sales')
plt.title('Time Series')
plt.show()
#画自相关图 和偏自相关图
plot_acf(data['NO.8']).show()
plot_pacf(data['NO.8']).show()
#平稳性检测
print('ADF result is',ADF(data['NO.8']))
#ADF result is (-4.185335160040106, 0.0006975913679687867, 4, 85, 
#{'1%': -3.5097356063504983, '5%': -2.8961947486260944, '10%': -2.5852576124567475}, 212.10556994032333)
#可以看出这里的ADF结果为-4.3， p-value小于0.05,  1%的结果-3.45>ADF的结果-4.3，
#所以说明极其显著拒绝原假设，不存在单位根，所以可以得出结论本数据是平稳的。

#纯随机检查
print('纯随机检查结果为',acorr_ljungbox(data['NO.8'],lags=1)) #lags表示延迟期数
#可以看出延迟1期后，p值为3.87e-19小于0.05，因此95%可以拒绝原假设，认为该序列为非白噪声序列，有信息可以提取

sm.graphics.tsa.plot_acf(data['NO.8'], lags=50)
plt.figure(figsize=(12, 6))
plt.show()

from statsmodels.tsa.stattools import arma_order_select_ic
print('BIC求解的模型阶次为',arma_order_select_ic(file.DO,max_ar=4,max_ma=2,ic='bic')['bic_min_order'])