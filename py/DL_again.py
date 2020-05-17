from __future__ import print_function
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
#import tensorflow.compat.v1 as tf
import  tensorflow as tf
tf.compat.v1.disable_eager_execution()

# 1.训练的数据
# Make up some real data
data= pd.read_csv("D:\pycharm项目\data.csv")
X = data.drop(data.columns[87],axis=1)  #特征集合X是除去最后一列标签的数据集
Y = data.iloc[:,[87]]  #标签集合Y是最后一列标签形成的数据集

X = X.drop(['Unnamed: 0','Flow ID',' Source IP',' Timestamp',' Destination IP',' Fwd Header Length','Flow Bytes/s',' Flow Packets/s','SimillarHTTP'],axis=1)
#Y= Y.astype('category')
X = X.values #将Pandas中的DataFrame转换成Numpy中数组
Y = Y.values


#for i in range(0, len(Y)):  # 将数据集分成攻击和良性两类
   # if Y[i] == 'UDP':
     #   continue
   # if Y[i] == 'UDPLag':
   #     continue
    #if Y[i] == 'NetBIOS':
   #     continue
   # if Y[i] == 'Syn':
   #     continue
   # if Y[i] == 'Portmap':
   #     continue
   # if Y[i] == 'LDAP':
    #    continue
    #if Y[i] == 'MSSQL':
   #     continue
   # else:
    #    Y[i] = 'other'


#for i in range(0, len(Y)):  #将多标签对应成数字
 #      Y[i] = 1
 #   elif Y[i] == 'UDPLag':
 #       Y[i] = 2
  #  elif Y[i] == 'NetBIOS':
  #      Y[i] = 3
  #  elif Y[i] == 'Syn':
 #       Y[i] = 4
  #  elif Y[i] == 'Portmap':
 #       Y[i] = 5
   # elif Y[i] == 'LDAP':
  #      Y[i] = 6
  #  elif Y[i] == 'MSSQL':
 #       Y[i] = 7
   # elif Y[i] == 'BENIGN':
#        Y[i] = 0
  #  else:
  #      Y[i] = -1
for i in range(0, len(Y)):  #将多标签对应成两种数字
    if Y[i] == 'UDP':
        Y[i] = 1
    elif Y[i] == 'UDPLag':
        Y[i] = 1
    elif Y[i] == 'NetBIOS':
        Y[i] = 1
    elif Y[i] == 'Syn':
        Y[i] = 1
    elif Y[i] == 'Portmap':
        Y[i] = 1
    elif Y[i] == 'LDAP':
        Y[i] = 1
    elif Y[i] == 'MSSQL':
        Y[i] = 1
    elif Y[i] == 'BENIGN':
        Y[i] = 0
    else:
        Y[i] = -1
#np.set_printoptions(threshold = np.inf)#打印整个数组
#print (Y)
def add_layer(inputs, in_size, out_size, activation_function=None):
   # add one more layer and return the output of this layer
   Weights = tf.Variable(tf.random.normal([in_size, out_size]))
   biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
   Wx_plus_b = tf.matmul(inputs, Weights) + biases
   if activation_function is None:
       outputs = Wx_plus_b
   else:
       outputs = activation_function(Wx_plus_b)
   return outputs
# 2.定义节点准备接收数据
# define placeholder for inputs to network
xs = tf.compat.v1.placeholder(tf.float32,[145643,78])
ys = tf.compat.v1.placeholder(tf.float32,[145643,1])


# 3.定义神经层：隐藏层和预测层
# add hidden layer 输入值是 xs，在隐藏层有 3个神经元
l1 = add_layer(xs,78, 5, activation_function = tf.nn.softmax)
# add output layer 输入值是隐藏层 l1，在预测层输出 1个结果
prediction = add_layer(l1,5,1,activation_function = None)

# 4.定义 loss 表达式
# the error between prediciton and real data
loss = tf.compat.v1.reduce_mean(tf.reduce_sum(tf.square(ys - prediction),axis=[1]))
#cross_entropy = tf.compat.v1.reduce_mean(-tf.reduce_sum(ys * tf.compat.v1.log(prediction),axis=[1]))
# 5.选择 optimizer 使 loss 达到最小
# 这一行定义了用什么方式去减少 loss，学习率是 0.1
train_step = tf.compat.v1.train.GradientDescentOptimizer(0.1).minimize(loss)


# important step 对所有变量进行初始化
init = tf.compat.v1.global_variables_initializer()
sess = tf.compat.v1.Session()
# 上面定义的都没有运算，直到 sess.run 才会开始运算
sess.run(init)
# 迭代 800次学习，sess.run optimizer
for i in range(800):
   # training train_step 和 loss 都是由 placeholder 定义的运算，所以这里要用 feed 传入参数
   sess.run(train_step, feed_dict={xs:X, ys:Y})
   if i % 40== 0:
       # to see the step improvement
       print(sess.run(loss, feed_dict={xs:X, ys:Y}))