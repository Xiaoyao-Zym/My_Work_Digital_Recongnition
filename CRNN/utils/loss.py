# import matplotlib.pyplot as plt
# import numpy as np
# import time


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


def loss(isTrue, str_time):
    log_data = pd.read_csv('log/'+str_time+'.csv')  # 打开csv文件
    train_loss_list = []
    train_loss_list=log_data.loc[:, 'train_loss']
    train_acc_list = []
    train_acc_list=log_data.loc[:, 'train_acc']
    val_loss_list = []
    val_loss_list=log_data.loc[:, 'val_loss']
    val_acc_list = []
    val_acc_list=log_data.loc[:, 'val_acc']
    num_list = []
    num_list=log_data.loc[:, 'step']
    plt.semilogy(num_list, train_loss_list, ls='-', color='cornflowerblue')  # 绘制x,y的折线图
    plt.semilogy(num_list, train_acc_list, ls='-', color='cornflowerblue')  
    plt.semilogy(num_list, val_loss_list, ls='-.', color='magenta')  # 绘制x,y的折线图
    plt.semilogy(num_list, val_acc_list, ls='-.', color='magenta')  
    plt.legend(labels=['train_loss', 'train_acc', 'val_loss', 'val_acc'], loc='best')
    plt.xlabel('epoch')
    plt.ylabel('loss/acc')
    plt.title('Training Curve')
    path='log/output/'
    if isTrue:
        plt.savefig(path+str_time+'_'+ str("{:.2f}".format(val_acc_list[len(val_acc_list)-1]))+'.jpg')
        plt.close()
        os.remove('log/output/temp.jpg')
    else:
        plt.savefig('log/output/temp.jpg')
        plt.close()