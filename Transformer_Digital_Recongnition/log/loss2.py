import os
import time
import pandas as pd
import matplotlib.pyplot as plt

def loss_map(isTrue, file_path):
    #log_data = pd.read_csv('./log/train_record/' + str_time + '.csv')  # 打开csv文件
    str_time=time.strftime('%Y-%m-%d_', time.localtime())
    root_path =os.path.join(os.path.realpath(os.curdir), 'log', 'loss') #获取当前目录的绝对路径
    # print(root_path)
    path2=os.path.join(root_path, "temp2.jpg") #
    log_data = pd.read_csv(file_path)  # 打开csv文件
    train_loss= []
    train_loss= log_data.loc[:, 'train_loss']
    valid_loss = []
    valid_loss = log_data.loc[:, 'valid_loss']
    train_list = []
    train_list = log_data.loc[:, 'step']
    plt.subplot(2, 1, 1)
    plt.plot(train_list, train_loss, ls='-', color='orange',
                 marker="o")  # 绘制x,y的折线图
    plt.legend("train_loss")
    
    plt.subplot(2, 1, 2)
    plt.plot(train_list, valid_loss, ls='-', color='b', marker="h")
    plt.xlabel("step")
    plt.legend("valid_loss")
    
    file_image= str_time + '_' +str("{:.3f}".format(valid_loss[len(valid_loss) - 1])) +'.jpg'
    image_path=os.path.join(root_path, file_image)
    if isTrue:
        plt.savefig(image_path)
        plt.show()
        os.remove(path2)
    else:
        plt.savefig(path2)
        plt.close()