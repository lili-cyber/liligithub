import pandas as pd
data= pd.read_csv("D:\pycharm项目\data.csv")
X = data.drop(data.columns[87],axis=1)  #特征集合X是除去最后一列标签的数据集
Y = data.iloc[:,[87]]
print(X.corr())#任意两列的相关系数
def pearsonr():#两两特征之间进行特征值计算
    factor = ['Source Port', 'Destination Port', 'Protocol', 'Flow Duration', 'Total Fwd Packets','Total Backward Packets', 'Total Length of Fwd Packets', 'Total Length of Bwd Packets', 'Fwd Packet Length Max', 'Fwd Packet Length Min','Fwd Packet Length Mean', 'Fwd Packet Length Std ','Bwd Packet Length Max','Bwd Packet Length Min','Bwd Packet Length Mean','Bwd Packet Length Std','Flow Bytes/s','Flow Packets/s','Flow IAT Mean', 'Flow IAT Std','Flow IAT Max','Flow IAT Min','Fwd IAT Total','Fwd IAT Mean',' Fwd IAT Std','Fwd IAT Max','Fwd IAT Min',' Bwd IAT Total','Bwd IAT Mean','Bwd IAT Std','Bwd IAT Max','Bwd IAT Min','Fwd PSH Flags','Bwd PSH Flags','Fwd URG Flags','Bwd URG Flags','Fwd Header Length','Bwd Header Length','Fwd Packets/s', 'Bwd Packets/s','Min Packet Length','Max Packet Length','Packet Length Mean','Packet Length Std',' Packet Length Variance','FIN Flag Count','SYN Flag Count','RST Flag Count','PSH Flag Count','ACK Flag Count','URG Flag Count','CWE Flag Count','ECE Flag Count','Down/Up Ratio','Average Packet Size','Avg Fwd Segment Size','Avg Bwd Segment Size','Fwd Header Length.1','Fwd Avg Bytes/Bulk','Fwd Avg Packets/Bulk','Fwd Avg Bulk Rate','Bwd Avg Bytes/Bulk','Bwd Avg Packets/Bulk','Bwd Avg Bulk Rate','Subflow Fwd Packets','Subflow Fwd Bytes','Subflow Bwd Packets','Subflow Bwd Bytes','Init_Win_bytes_forward','Init_Win_bytes_backward','act_data_pkt_fwd','min_seg_size_forward','Active Mean','Active Std','Active Max','Active Min','Idle Mean','Idle Std','Idle Max','Idle Min','Inbound']
    for i in range (len(factor)):
        for j in range(i,len(factor)-1):
            print ("指标%s与指标%s之间的相关性大小为%f" % (factor[i],factor[j]))
    return None
