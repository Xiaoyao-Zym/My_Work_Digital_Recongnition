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
    train_loss= []
    train_loss= log_data.loc[:, 'train_loss']
    valid_loss = []
    valid_loss = log_data.loc[:, 'train_acc']

    num_list = []
    num_list = log_data.loc[:, 'step']
    plt.semilogy(num_list, train_loss, ls='-', color='orange',
                 marker="o")  # 绘制x,y的折线图
    plt.semilogy(num_list, valid_loss, ls='-', color='pink', marker="o")

    plt.legend(labels=['train_loss', 'valid_loss'],
               loc='best')
    plt.xlabel('epoch')
    plt.ylabel('loss/acc')
    plt.title('Training Curve')
    file_image= str_time + '_' +str("{:.2f}".format(valid_loss[len(valid_loss) - 1])) +'.jpg'
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