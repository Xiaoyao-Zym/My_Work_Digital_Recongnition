# import matplotlib.pyplot as plt
# import numpy as np
from logging import root
import time

# def Result_plot(isSave, x_vals, y_vals,
#                 x_label, y_label,
#                 x2_vals=None, y2_vals=None,
#                 legend=None,
#                 figsize=(3.5, 2.5)):
#     # set figsize
#    try:
#       train_loss_lines.remove(train_loss_lines[0]) #移除上一步曲线
#       val_acc_lines.remove(val_acc_lines[0])
#    except Exception:
#     pass
#     plt.xlabel(x_label)
#     plt.ylabel(y_label)
#     train_loss_lines=plt.semilogy(x_vals, y_vals)
#     if x2_vals and y2_vals:
#       val_acc_lines=plt.semilogy(x2_vals, y2_vals, linestyle=':')

#     if legend:
#         plt.legend(legend)
#     plt.pause(0.5)
#     if isSave:
#          str_time=time.strftime('%Y-%m-%d_', time.localtime())
#          path='log/output/'
#        plt.savefig(path+str_time+ str("{:.2f}".format(y2_vals[-1]))+'.jpg', bbox_inches='tight')
import os
import pandas as pd
import matplotlib.pyplot as plt

def loss(isTrue, data_path):
    #log_data = pd.read_csv('./log/train_record/' + str_time + '.csv')  # 打开csv文件
    str_time=time.strftime('%Y-%m-%d_', time.localtime())
    root_path =os.path.join(os.path.realpath(os.curdir), 'log', 'loss') #获取当前目录的绝对路径
    print(root_path)
    path2=os.path.join(root_path, "temp.jpg") #
    log_data = pd.read_csv(data_path)  # 打开csv文件
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
    plt.semilogy(num_list, train_acc_list, ls='-', color='cyan', marker="o")
    plt.semilogy(num_list, val_loss_list, ls='-.', color='blue',
                 marker="d")  # 绘制x,y的折线图
    plt.semilogy(num_list, val_acc_list, ls='-.', color='pink', marker="d")
    plt.legend(labels=['train_loss', 'train_acc', 'val_loss', 'val_acc'],
               loc='best')
    plt.xlabel('epoch')
    plt.ylabel('loss/acc')
    plt.title('Training Curve')
    file_image= str_time + '-' +str("{:.2f}".format(val_acc_list[len(val_acc_list) - 1])) +'.jpg'
    image_path=os.path.join(root_path, file_image)
    if isTrue:
        plt.savefig(image_path)
        plt.show()
        os.remove(path2)
    else:
        plt.savefig(path2)
        plt.close()
        
# if __name__=='_main_':
#     str_time=time.strftime('%Y-%m-%d', time.localtime())
#     loss(str_time=str_time, isTrue=False)