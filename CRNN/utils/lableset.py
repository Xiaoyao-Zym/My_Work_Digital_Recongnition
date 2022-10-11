import pandas as pd
import os

file_path="data/label.txt"#文件目录
data=pd.read_csv(file_path, sep='\t', names=['image', 'lable'], encoding="UTF-8") # 读入
d1=data['image'].str.split('/', expand=True)  #以/分割
d1.columns = ['无用','image']
data_new=d1.drop('无用',axis=1).join(data.drop('image',axis=1)) #删除无用列，合并有用列
data_new.to_csv(file_path, mode='w', sep=' ',  header=False, index=False) #写入原文件