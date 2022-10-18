import os,time
from markdown import markdown
import pandas as pd
import matplotlib.pyplot as plt

isTure=True
str_time=time.strftime('%Y-%m-%d', time.localtime())
log_data = pd.read_csv('../log/' + str_time + '.csv')  # 打开csv文件
#log_data = pd.read_csv('../log/2022-10-18.csv')  # 打开csv文件
train_loss_list = []
train_loss_list = log_data.loc[:, 'train_loss']
train_acc_list = []
train_acc_list = log_data.loc[:, 'train_acc']
val_loss_list = []
val_loss_list = log_data.loc[:, 'val_loss']
val_acc_list = []
val_acc_list = log_data.loc[:, 'val_acc']
num_list = []
num_list = log_data.loc[:, 'step']
plt.semilogy(num_list, train_loss_list, ls='-', color='orange',
                marker="o")  # 绘制x,y的折线图
plt.semilogy(num_list, train_acc_list, ls='-', color='pink', marker="o")
plt.semilogy(num_list, val_loss_list, ls='-.', color='orange',
                marker="d")  # 绘制x,y的折线图
plt.semilogy(num_list, val_acc_list, ls='-.', color='pink', marker="d")
plt.legend(labels=['train_loss', 'train_acc', 'val_loss', 'val_acc'],
            loc='best')
plt.xlabel('epoch')
plt.ylabel('loss/acc')
plt.title('Training Curve')
path = '../log/output/'
if  isTure:
    plt.savefig(path + str_time + '_' +
                str("{:.2f}".format(val_acc_list[len(val_acc_list) - 1])) +
                '.jpg')
    plt.show()
    #os.remove('../log/output/temp.jpg')
else:
    plt.savefig('../log/output/temp.jpg')
    plt.close()