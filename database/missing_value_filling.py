from  sklearn import ensemble
from sklearn.preprocessing import LabelEncoder
import pandas as pd

def set_missing(df,estimate_list,miss_col):
    """df要处理的数据帧，estimate_list输入的字段名称,miss_col缺失字段名称;会直接在原来的数据帧上修改"""
    col_list = []
    col_list.extend(estimate_list)
    col_list.append(miss_col)
    process_df = df.loc[:  , col_list]      # 所有列
    # class_le = LabelEncoder()
    # for i in col_list[:-1]:                       # 最后一个以前, 输入字段名称
    #     print('col_list[:-1]',col_list[:-1])
    #     process_df.loc[:,i] = class_le.fit_transform(process_df.loc[:,i].values)
    # 分成已知该特征和未知该特征两部分
    known = process_df[process_df[miss_col].isnull() & process_df[estimate_list[0]].notnull()].values   # 预测并返回集 要预测的列为空，用于预测的列非空  注意  .value
    # known[:, -1]=class_le.fit_transform(known[:, -1])
    unknown = process_df[process_df[miss_col].isnull() & process_df[estimate_list[0]].notnull()]
    all_known = process_df[process_df[miss_col].notnull() & process_df[estimate_list[0]].notnull()].values  # 训练集 要预测的为不为空，用于预测的列也不空

    all_known_x = all_known[:, :-1]    # 训练集x， 用于预测的列
    all_known_y = all_known[:, -1]    # 训练集y， 要预测的列
    # X为特征属性值
    x = known[:, :-1] # 输入字段
    # y为结果标签值
    y = known[:, -1]  # 缺失字段

    # fit到RandomForestRegressor之中
    rfr = ensemble.RandomForestRegressor(random_state=1, n_estimators=1000,max_depth=8,n_jobs=-1)
    rfr.fit(all_known_x,all_known_y)
    score=rfr.score(all_known_x,all_known_y)
    print(score)
    # 用得到的模型进行未知特征值预测
    predicted = rfr.predict(x).round(0).astype(int)
    # predicted = class_le.inverse_transform(predicted)
    # print(predicted)
    # 用得到的预测结果填补原缺失数据
    df.loc[(df[miss_col].isnull() & df[estimate_list[0]].notnull()), miss_col] = predicted   # 缺失值为空，用于预测的值非空
    return df

if __name__=='__main__':
    df = pd.read_csv('C:/Users/41634/Desktop/WMS_1440.csv')
    #df=df.drop(['DATETIME'],axis=1)
    columns=df.columns
    print(columns[1])
    print(columns[2])
    #dff=set_missing(df,columns[:, 1],columns[:, 2])
    #dff=set_missing(df,['F_NH3'],'F_TP')
    dff = set_missing(df, ['F_TN'], 'F_TP')
    dff.to_csv('C:/Users/41634/Desktop/1440.csv')