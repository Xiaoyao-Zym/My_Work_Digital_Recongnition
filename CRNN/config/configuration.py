from dataclasses import field
import sys
import os
#加上下面代码
os.chdir(sys.path[0])
import torch
import torch.nn as nn
import argparse
from model.crnn import CRNN
from log.log import *
from train_batch import train_batch
from utils.aftertreatment import StrLabelConverter
from dataset.datasets import get_DataLoader
import time
from log.loss import loss

def parse_opt():
    parser = argparse.ArgumentParser(description='train')
    parser.add_argument('--epochs', type=int, default=50, help='训练多少轮')
    parser.add_argument('--batch_size', type=int, default=32, help='批次大小')
    parser.add_argument('--lr', type=float, default=0.001, help='学习率')
    parser.add_argument('--chinese', type=str, default='./labels/figure.txt', help='字符集保存路径')
    parser.add_argument('--images', type=str, default='../data/image/', help='你可以设置你所以图片的地址，像现在的默认值，也可以设置为data/images/'
                                                                              '这样你就需要在这个目录下要有训练集，验证集，测试集的图片，可以运行'
                                                                              'dataset/splitimages.py生成，但不建议使用第二种方式')
    parser.add_argument('--labels', type=str, default='./labels/', help='标签的路径')
    parser.add_argument('--imgH', type=int, default=32)
    parser.add_argument('--nc', type=int, default=1)
    parser.add_argument('--nh', type=int, default=251)
    parser.add_argument('--val_epoch', type=int, default=1, help='经过多少个epoch验证一次')
    parser.add_argument('--save_all', action='store_true', default=True, help='是否保存所有模型')
    parser.add_argument('--best', type=float, default=0.9, help='如果不保存所有模型，他就之会保存最好的和最后的模型，最好的模型准确率必须高于best才会保存')
    parser.add_argument('--test', action='store_true', default=False, help='模型训练好是否测试')
    parser.add_argument('--all', action='store_true', default=False,
                        help='如果开启测试，False的话就只会输出预测错误的，True就不管预测正确还是错误，都会输出')
    parser.add_argument('--weights', type=str, default='', help='如果因为种种原因导致训练停止，但保存了模型，可以从这个模型开始训练，填入模型的路径')
    parser.add_argument('--name', type=str, default='./weights/', help='模型保存的位置')
    parser.add_argument('--log', type=str, default='./log/', help='模型保存的位置')
    opt = parser.parse_args()
    return opt