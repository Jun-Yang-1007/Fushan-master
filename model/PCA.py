import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import pandas as pd
dataframe = pd.read_csv('../newdata.csv')
data = dataframe.iloc[0:100, 2:8]

pca = PCA(n_components = 2)
pca = pca.fit_transform(data )
print(pca)


import numpy as np
import pandas
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
dataframe = pandas.read_csv('E:/MyFpi/Project1/fushan/Data/section_1_day_data.csv')
data = dataframe.iloc[0:300,2:8]
data = data.dropna()
pca_line = PCA().fit(data)
plt.plot([1,2,3,4,5,6],np.cumsum(pca_line.explained_variance_ratio_))
plt.xticks([1,2,3,4,5,6]) #这是为了限制坐标轴显示为整数
plt.xlabel("number of components after dimension reduction")
plt.ylabel("cumulative explained variance")
plt.show()

pca = PCA(n_components=2)
pca = pca.fit(data)
X_dr = pca.transform(data)
print(pca.explained_variance_ratio_,pca.explained_variance_ratio_.sum())