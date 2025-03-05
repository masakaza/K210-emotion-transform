import pandas as pd
import os
from sklearn.preprocessing import OneHotEncoder
wav_pth='./casia/'
csv_file_pth='labels.csv'

labels_csv=pd.read_csv(csv_file_pth)

wav_file=[ wav_pth+x+'/' for x in os.listdir(wav_pth)]
labels_class=[ [x] for x in os.listdir(wav_pth) ]

encoder= OneHotEncoder()
one_hot_label = encoder.fit_transform(labels_class).toarray()

print(wav_file)   #打印文件下级目录

for i in range(len(wav_file)):
    wav_name=os.listdir(wav_file[i])     #获取下级目录中的每一个文件名称
    wav_pth_file=[ wav_file[i]+x for x in wav_name ]       #合并目录
    labels_one_hot=[ list(one_hot_label[i]) for x in range(len(wav_pth_file)) ]   #获取每个文件对应的one-hot标签
    write_data=pd.DataFrame({
        'filename': wav_pth_file,
        'label':labels_one_hot,
    })
    labels_csv=pd.concat([labels_csv,write_data],ignore_index=True)   #合并内容
    labels_csv.to_csv(csv_file_pth,mode='w',index=False,header=True)
