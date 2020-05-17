# -*- coding: UTF-8 -*-
import numpy as np
import pandas as pd
import linecache
import matplotlib.pyplot as plt

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import precision_recall_fscore_support
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import scale
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.externals import joblib



data= pd.read_csv("D:\pycharm项目\data.csv")
X = data.drop(data.columns[87],axis=1)  #特征集合X是除去最后一列标签的数据集
Y = data.iloc[:,[87]]  #标签集合Y是最后一列标签形成的数据集

X=X.drop(['Unnamed: 0','Flow ID',' Source IP',' Timestamp',' Destination IP',' Fwd Header Length','Flow Bytes/s',' Flow Packets/s','SimillarHTTP'],axis=1)
#Y= Y.astype('category')
X=X.values
Y=Y.values  #将Pandas中的DataFrame转换成Numpy中数组

for i in range(0, len(Y)):  # 将值'UDP'替换为0，将其他标签替换为1
    if Y[i] == 'UDP':
        continue
    if Y[i] == 'UDPLag':
        continue
    if Y[i] == 'NetBIOS':
        continue
    if Y[i] == 'Syn':
        continue
    if Y[i] == 'Portmap':
        continue
    if Y[i] == 'LDAP':
        continue
    if Y[i] == 'MSSQL':
        continue
    else:
        Y[i] = 'other'



X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2)

cls = {
    '决策树': DecisionTreeClassifier(max_depth=20),
    '随机森林': RandomForestClassifier(n_estimators=150,random_state=2),
    '贝叶斯':GaussianNB(),
    'GBDT':GradientBoostingClassifier(random_state=10,subsample=0.8,max_features='log2'),
    '神经网络': MLPClassifier(hidden_layer_sizes = (64,3)),
    '投票分类器':VotingClassifier(estimators=[('rf',RandomForestClassifier(n_estimators=150,random_state=2)),
                                         ('gbdt',GradientBoostingClassifier(random_state=10,subsample=0.8,max_features='log2')),
                                         ('nns',MLPClassifier(hidden_layer_sizes = (64,3)))],voting='soft')
    }

for i in cls:
    print('分类器为:', i)
    model = Pipeline([
        ('cl', cls[i])#双分类问题ss
    ])
    model.fit(X_train, Y_train)
    pre_Y_test = model.predict(X_test)
#评价
    accuracy = accuracy_score(Y_test, pre_Y_test)
    precision,recall,f1,support = precision_recall_fscore_support(Y_test, pre_Y_test)
    print('准确率为:',accuracy,'\n精确率为:',precision,'\n召回率为:',recall,'\nF1为:',f1)
    joblib.dump(model,'machine multify.pkl')