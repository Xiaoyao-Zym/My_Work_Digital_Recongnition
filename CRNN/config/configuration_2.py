import sys
import os
#加上下面代码
os.chdir(sys.path[0])
import argparse

def parse_opt():
    parser = argparse.ArgumentParser(description='train')
    parser.add_argument('--epochs', type=int, default=30, help='训练多少轮')
    parser.add_argument('--batch_size', type=int, default=64, help='批次大小')
    parser.add_argument('--lr', type=float, default=0.001, help='学习率')
    parser.add_argument('--chinese', type=str, default='./labels/figure.txt', help='字符集保存路径')
    parser.add_argument('--images', type=str, default='../data/image', help='你可以设置你所以图片的地址，像现在的默认值，也可以设置为data/images/'
                                                                              '这样你就需要在这个目录下要有训练集，验证集，测试集的图片，可以运行'
                                                                              'dataset/splitimages.py生成，但不建议使用第二种方式')
    parser.add_argument('--labels', type=str, default='./labels/', help='标签的路径')
    parser.add_argument('--imgH', type=int, default=32)
    parser.add_argument('--nc', type=int, default=1)
    parser.add_argument('--nh', type=int, default=256)
    parser.add_argument('--num_class', type=int, default=11)
    parser.add_argument('--hidden_size', type=int, default=256, help='隐藏层数')
    parser.add_argument('--val_epoch', type=int, default=1, help='经过多少个epoch验证一次')
    parser.add_argument('--save_all', action='store_true', default=True, help='是否保存所有模型')
    parser.add_argument('--best', type=float, default=0.9, help='如果不保存所有模型，他就之会保存最好的和最后的模型，最好的模型准确率必须高于best才会保存')
    parser.add_argument('--test', action='store_true', default=False, help='模型训练好是否测试')
    parser.add_argument('--all', action='store_true', default=False, help='如果开启测试，False的话就只会输出预测错误的，True就不管预测正确还是错误，都会输出')
    parser.add_argument('--weights', type=str, default='', help='如果因为种种原因导致训练停止，但保存了模型，可以从这个模型开始训练，填入模型的路径')
    parser.add_argument('--name', type=str, default='./weights/', help='模型保存的位置')
    parser.add_argument('--log', type=str, default='./log/', help='模型保存的位置')
    parser.add_argument('--beta1', type=float, default=0.9, help='beta1 for adam. default=0.9')
    parser.add_argument('--rho', type=float, default=0.95, help='decay rate rho for Adadelta. default=0.95')
    parser.add_argument('--character', type=str, default='0123456789', help='character label')
    parser.add_argument('--eps', type=float, default=1e-8, help='eps for Adadelta. default=1e-8')
    parser.add_argument('--Transformation', default='None' ,type=str, help='Transformation stage. None|TPS')
    parser.add_argument('--FeatureExtraction', type=str, default='DRNet_V1', help='FeatureExtraction stage. VGG|RCNN|ResNet|DRNet_V1')
    parser.add_argument('--SequenceModeling', type=str, default='BiLSTM', help='SequenceModeling stage. None|BiLSTM')
    parser.add_argument('--Prediction', type=str, default='Transformer', help='Prediction stage. CTC|Attn|Transformer')
    parser.add_argument('--imgW', type=int, default=280, help='the width of the input image')
    parser.add_argument('--num_fiducial', type=int, default=20, help='number of fiducial points of TPS-STN')
    parser.add_argument('--input_channel', type=int, default=1,  help='the number of input channel of Feature extractor')          
    parser.add_argument('--output_channel', type=int, default=512, help='the number of output channel of Feature extractor')    
    parser.add_argument('--optimizer', default='adam', help='Whether to use adam (default is Adadelta)')
               
    return parser.parse_args(args=[])