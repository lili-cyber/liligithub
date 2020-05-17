#coding=utf-8
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import  DecisionTreeClassifier
from sklearn.tree import export_graphviz
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import StandardScaler
from pandas import DataFrame



import csv
#data = pd.read_csv("E:\pycharm项目\syn.csv",low_memory=False)
#以若干行为块读取
def get_df(file):
    mylist = []
    for chunk in pd.read_csv(file, chunksize=10000):
        mylist.append(chunk)
    temp_df = pd.concat(mylist,axis = 0,ignore_index=True)
    del mylist
    return temp_df
file = r'E:\pycharm项目\syn.csv'
data = get_df(file)
data.head()
#数据处理
#1）任意选择行数
#sample = data.sample(n=3000,replace=None,axis=0)
#sample.to_csv("E:\pycharm项目\datasyn.csv",encoding='utf_8_sig')
#2)选择特征值
#x= data.drop('Label',axis=1)#特征集合x是除去最后一列标签的数据集
#y_new = data['Label']
#y = y_new.astype('category')#标签转换成类
#转换成字典
#x.to_dict(orient="records")

#3)划分数据集
#x_train,y_train,x_test,y_test = train_test_split(x,y,test_size=0.2,random_state=22)
#字典特征抽取
#实例化一个转换器类
#transfer = DictVectorizer(sparse=False)
#x_train = transfer.fit_transform(x_train)
#x_test = transfer.fit_transform(x_test)

#4）决策器预估器
#estimator = DecisionTreeClassifier(criterion="entropy")
#estimator.fit(x_train,y_train)
#5)模型评估
#第一种方法：直接比较真实值和预测值、
#y_predict = estimator.predict(x_test)
#print("y_predict:\n",y_predict)
#print("直接比对真实值和预测值:\n",y_test = y_predict)
#方法二：计算准确率
#score = estimator.score(x_test,y_test)
#print("准确率为：\n",score)
#export_graphviz(estimator,out_file="E:\pycharm项目\syn_simple_tree.dot",feature_names=transfer.get_feature_names())










