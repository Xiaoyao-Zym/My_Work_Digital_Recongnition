import os
import torch
from log.log import *
from train_batch_DR import train_batch
from dataset.datasets import get_DataLoader
import time
import tqdm
from val_DR import val_DR
from test import test
import pandas as pd
import numpy as np
import time
from log.loss import loss
from config.configuration_2 import parse_opt
from modules.utils import CTCLabelConverter, CTCLabelConverterForBaiduWarpctc, AttnLabelConverter, TransformerLabelConverter, Averager
from model.DRNet_V2 import  DRNet_V2
#from model.mobilenetv3 import MobileNetV3_Small
#device = torch.device("cuda:0")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train(opt):
    """ dataset preparation """
    if not os.path.exists(opt.name):
        os.makedirs(opt.name)
        
    #定义训练日志保存目录
    str_time=time.strftime('%Y-%m-%d', time.localtime())
    df = pd.DataFrame(columns=['time', 'step', 'train_loss', 'train_acc', 'val_loss', 'val_acc'])#列名
    file_dir=opt.log+'/train_record/DRnet_V2/'
    if os.path.isdir(file_dir):
        pass
    else:
        os.makedirs(file_dir)
        
    #创建模型保存目录
    model_dir =opt.name+'DRnet_V2/'+str_time+ '/'
    if os.path.isdir(model_dir):
        pass
    else:
        os.makedirs(model_dir)  
    
    file_path=file_dir+str_time+'.csv'
    df.to_csv(file_path, index=False) #路径可以根据需要更改
    s_time = time.time()
    # nclass = len(chinese) + 1
    best_model = {}
    best = opt.best
    #加载数据
    train_loader = get_DataLoader('train', opt) # 训练集
    val_loader = get_DataLoader('val', opt) # 验证集
    #字符编码
    #character = get_figure(opt.chinese)
    #converter = StrLabelConverter(figure)
    
    """ model configuration """
    if 'CTC' in opt.Prediction:
        if opt.baiduCTC:
            converter = CTCLabelConverterForBaiduWarpctc(opt.character)
        else:
            converter = CTCLabelConverter(opt.character)
    elif "Transformer" in opt.Prediction:
        converter = TransformerLabelConverter(opt.character)
       #converter=StrLabelConverter(opt.character)
    else:
        converter = AttnLabelConverter(opt.character)
    #opt.num_class = len(converter. character)
    model=DRNet_V2(opt).to(device)
    #opt.num_class = len(converter.character)
    # weight initialization
    if os.path.exists(opt.weights):
        model.load_state_dict(torch.load(opt.weights))
    log_load_model(opt.weights)
   
    """ setup loss """
    if 'CTC' in opt.Prediction:
        criterion = torch.nn.CTCLoss(zero_infinity=True)
    elif "Transformer" in opt.Prediction:
        # criterion = torch.nn.CrossEntropyLoss(ignore_index=2).to(device)  # ignore [PAD] token = ignore index 1
        criterion = torch.nn.CTCLoss(zero_infinity=True)
    else:
        criterion = torch.nn.CrossEntropyLoss(ignore_index=0)  # ignore [GO] token = ignore index 0
        
    #loss_avg = Averager()
    # filter that only require gradient decent
    filtered_parameters = []
    params_num = []
    for p in filter(lambda p: p.requires_grad, model.parameters()):
        filtered_parameters.append(p)
        params_num.append(np.prod(p.size()))
    print('Trainable params num : ', sum(params_num))
    
    # setup optimizer
    if opt.optimizer=='adam':
        optimizer = torch.optim.Adam(filtered_parameters, lr=opt.lr, betas=(opt.beta1, 0.999))
    else:
        print("use Adadelta")
        optimizer = torch.optim.Adadelta(filtered_parameters, lr=opt.lr, rho=opt.rho, eps=opt.eps)
    print("Optimizer:")
    print(optimizer)
    
    #记录参数
    log_parameter(opt, device)
    log_optimizer(optimizer)
    log_model(model)

    #开始训练
    for epoch in range(1, opt.epochs + 1):
        train_iter = iter(train_loader)
        val_iter = iter(val_loader)
        total_loss = 0.
        total_num = 0.
        total_acc = 0.

        with tqdm.tqdm(range(len(train_iter))) as tbar:
            for i in tbar:
                model.train()
                # train_loss, train_size,  = train_batch(model, train_iter, optimizer, criterion, device, converter)
                train_loss, train_size, acc_num = train_batch(model, train_iter, optimizer, criterion, device, converter)
                #train_loss, acc_num= loss_avg.add(train_loss)
                
                total_loss+=train_loss
                total_num += train_size
                total_acc += acc_num

                tbar.set_description('epoch {}'.format(epoch))
                tbar.set_postfix(loss=train_loss / train_size, acc=acc_num / train_size)
                tbar.update()

        
        log_epoch_val(total_loss / total_num, total_acc / total_num, 'train')
        
        train_los="{:.5f}".format (total_loss / total_num)
        train_ac= "{:.5f}".format(total_acc / total_num)
        timer = str_time
        step = epoch
                        
        if epoch % opt.val_epoch == 0:
            val_loss, val_acc = val_DR(model, val_iter, criterion, device, converter, epoch)
            list = [timer,step,train_los, train_ac, "{:.5f}".format(val_loss), "{:.5f}".format(val_acc)]
            data = pd.DataFrame([list])
            data.to_csv(file_path, mode='a',header=False,index=False)#mode设为a,就可以向csv文件追加数据了                
            if opt.save_all and val_acc>0.8:
                torch.save(crnn.state_dict(),
                          model_dir+str('%03d' %epoch) + '-train_loss' + str("{:.2f}".format(total_loss / total_num)) + '-val_loss' + str("{:.2f}".format(val_loss)) + '.pt')
                log_save_model(epoch, val_loss, val_acc)
                

            elif val_acc > opt.best:
                opt.best = val_acc
                best_model['epoch'] = epoch
                best_model['val_loss'] = val_loss
                best_model['val_acc'] = val_acc

            if epoch == opt.epochs and not opt.save_all:
                torch.save(crnn.state_dict(),
                            model_dir + 'last_'+str('%03d' %epoch) + '-_val_loss-' + str("{:.2f}".format(val_loss)) + '-val_acc' + str("{:.2f}".format(val_acc)) + '.pt')
                log_save_model(epoch, val_loss, val_acc, 'last')

            if(epoch==opt.epochs):
                isTure=True
                loss(isTure, file_path, model_type='DRnet_V2')
            else:
                isTure=False
                loss(isTure, file_path, model_type='DRnet_V2')
   
    if not opt.save_all and opt.best != best:
        torch.save(crnn.state_dict(),
                     model_dir+'best_'+ str(best_model['epoch']) + '_' + str(best_model['val_loss']) + '_' +str(best_model['val_acc']) + '.pt')
        
        log_save_model(best_model['epoch'], best_model['val_loss'], best_model['val_acc'], 'best')

    e_time = time.time()
    print('cost time:', round((e_time - s_time) / 3600., 2))

    if opt.test:
        if not opt.save_all:
            crnn = crnn.load_state_dict(
                torch.load(model_dir+'best_'+ str(best_model['epoch']) + '_' + str(best_model['val_loss']) + '_' +str(best_model['val_acc']) + '.pt'))
            
        test_loader = get_DataLoader('test', opt)
        test_iter = iter(test_loader)
        test(crnn, test_iter, criterion, device, converter, opt.all)


if __name__ == '__main__':
    opt = parse_opt()
    train(opt)
