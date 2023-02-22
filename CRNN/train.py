from dataclasses import field
import os
import torch
import torch.nn as nn
import argparse
from model.crnn import CRNN
from log.log import *
from train_batch import train_batch
from utils.aftertreatment import StrLabelConverter
from dataset.datasets import get_DataLoader
import time
import tqdm
from utils.fileoperation import get_chinese
from val import val
from test import test
import pandas as pd
import time
from log.loss import loss
from config.configuration import parse_opt

def train(opt):
    if not os.path.exists(opt.name):
        os.makedirs(opt.name)
    chinese = get_chinese(opt.chinese)
    converter = StrLabelConverter(chinese)
    nclass = len(chinese) + 1
    best_model = {}
    best = opt.best

    # 训练集
    train_loader = get_DataLoader('train', opt)
    # 验证集
    val_loader = get_DataLoader('val', opt)
    criterion = nn.CTCLoss(reduction='sum')
    crnn = CRNN(opt.imgH, opt.nc, nclass, opt.nh)

    if os.path.exists(opt.weights):
        crnn.load_state_dict(torch.load(opt.weights))
    log_load_model(opt.weights)

    optimizer = torch.optim.Adam(crnn.parameters(), lr=opt.lr)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    crnn = crnn.to(device)
    criterion = criterion.to(device)

    log_parameter(opt, device)
    log_optimizer(optimizer)
    log_model(crnn)

    str_time=time.strftime('%Y-%m-%d', time.localtime())
    df = pd.DataFrame(columns=['time', 'step', 'train_loss', 'train_acc', 'val_loss', 'val_acc'])#列名
    file_path=opt.log+'/train_record/'+str_time+'.csv'
    df.to_csv(file_path, index=False) #路径可以根据需要更改
    s_time = time.time()
    for epoch in range(1, opt.epochs + 1):
        train_iter = iter(train_loader)
        val_iter = iter(val_loader)
        total_loss = 0.
        total_num = 0.
        total_acc = 0.

        with tqdm.tqdm(range(len(train_iter))) as tbar:
            for i in tbar:
                crnn.train()
                train_loss, train_size, acc_num = train_batch(crnn, train_iter, optimizer, criterion, device, converter)

                total_loss += train_loss
                total_num += train_size
                total_acc += acc_num

                tbar.set_description('epoch {}'.format(epoch))
                tbar.set_postfix(loss=train_loss / train_size, acc=acc_num / train_size)
                tbar.update()

        
        log_epoch(epoch, total_loss / total_num, total_acc / total_num, 'train')
        train_los="{:.5f}".format (total_loss / total_num)
        train_ac= "{:.5f}".format(total_acc / total_num)
        timer = str_time
        step = epoch

        if epoch % opt.val_epoch == 0:
            val_loss, val_acc = val(crnn, val_iter, criterion, device, converter, epoch)
            list = [timer,step,train_los, train_ac, "{:.5f}".format(val_loss), "{:.5f}".format(val_acc)]
            data = pd.DataFrame([list])
            data.to_csv(file_path, mode='a',header=False,index=False)#mode设为a,就可以向csv文件追加数据了
            if opt.save_all and val_acc>0.5:
                torch.save(crnn.state_dict(),
                           opt.name+ str_time+ '/'+str('%03d' %epoch) + '-train_loss' + str("{:.2f}".format(total_loss / total_num)) + '-val_loss' + str("{:.2f}".format(val_loss)) + '.pt')
                log_save_model(epoch, val_loss, val_acc)
                

            elif val_acc > opt.best:
                opt.best = val_acc
                best_model['epoch'] = epoch
                best_model['val_loss'] = val_loss
                best_model['val_acc'] = val_acc

            if epoch == opt.epochs and not opt.save_all:
                torch.save(crnn.state_dict(),
                             opt.name+ str_time + '/last_'+str('%03d' %epoch) + '-_val_loss-' + str("{:.2f}".format(val_loss)) + '-val_acc' + str("{:.2f}".format(val_acc)) + '.pt')
                log_save_model(epoch, val_loss, val_acc, 'last')

            if(epoch==opt.epochs):
                isTure=True
                loss(isTure, file_path)
            else:
                isTure=False
                loss(isTure, file_path)

    if not opt.save_all and opt.best != best:
        torch.save(crnn.state_dict(),
                     opt.name+ str_time+'/best_'+ str(best_model['epoch']) + '_' + str(best_model['val_loss']) + '_' +
                   str(best_model['val_acc']) + '.pt')
        log_save_model(best_model['epoch'], best_model['val_loss'], best_model['val_acc'], 'best')

    e_time = time.time()
    print('cost time:', round((e_time - s_time) / 3600., 2))

    if opt.test:
        if not opt.save_all:
            crnn = crnn.load_state_dict(
                torch.load(opt.model_path+ str_time+'/best_'+ str(best_model['epoch']) + '_' + str(best_model['val_loss']) + '_' +
                           str(best_model['val_acc']) + '.pt'))
        test_loader = get_DataLoader('test', opt)
        test_iter = iter(test_loader)
        test(crnn, test_iter, criterion, device, converter, opt.all)


if __name__ == '__main__':
    opt = parse_opt()
    train(opt)
