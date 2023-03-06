#from torchsummary import summary
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from model.rnn import BidirectionalLSTM

class hswish(nn.Module):
    def forward(self, x):
        out = x * F.relu6(x + 3, inplace=True) / 6
        return out


class hsigmoid(nn.Module):
    def forward(self, x):
        out = F.relu6(x + 3, inplace=True) / 6
        return out

#注意力机制
class SeModule(nn.Module):
    def __init__(self, in_size, reduction=4):
        super(SeModule, self).__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_size, in_size // reduction, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(in_size // reduction),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_size // reduction, in_size, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(in_size),
            hsigmoid()
        )

    def forward(self, x):
        return x * self.se(x)

#线性瓶颈和反向残差结构
class Block(nn.Module):
    '''expand + depthwise + pointwise'''
    def __init__(self, kernel_size, in_size, expand_size, out_size, nolinear, semodule, stride):
        super(Block, self).__init__()
        self.stride = stride
        self.se = semodule

        self.conv1 = nn.Conv2d(in_size, expand_size, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(expand_size)
        self.nolinear1 = nolinear
        self.conv2 = nn.Conv2d(expand_size, expand_size, kernel_size=kernel_size, stride=stride, padding=kernel_size//2, groups=expand_size, bias=False)
        self.bn2 = nn.BatchNorm2d(expand_size)
        self.nolinear2 = nolinear
        self.conv3 = nn.Conv2d(expand_size, out_size, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_size)

        self.shortcut = nn.Sequential()
        if stride == 1 and in_size != out_size:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_size, out_size, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_size),
            )

    def forward(self, x):
        out = self.nolinear1(self.bn1(self.conv1(x)))
        out = self.nolinear2(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.se != None:
            out = self.se(out)
        out = out + self.shortcut(x) if self.stride==1 else out
        #print(out.shape)
        return out
      
    
class DRNet(nn.Module):
    def __init__(self,  imgH, nc, nclass, nh):
        """
        :param imgH: 图片高度
        :param nc: 图片通道数
        :param nclass: 类别个数
        :param nh: RNN中隐藏层神经元个数
        """
        
        super(DRNet, self).__init__()
        assert imgH % 16 == 0, '图片高度必须是16的倍数，建议32'
        
        self.conv1 = nn.Conv2d(in_channels=nc, out_channels=8, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(8)
        self.hs1 = hswish()
        
        self.blok=nn.Sequential(
            Block(3, 8, 16, 16, nn.ReLU(inplace=True), SeModule(16), 2), 
            #torch.Size([2, 16, 16, 140])
            Block(3, 16, 72, 32, nn.ReLU(inplace=True), None, (2,1)),   
            #torch.Size([2, 32, 8, 140])
            Block(3, 32, 88, 32, nn.ReLU(inplace=True), None, 1),   
            #torch.Size([2, 32, 8, 140])
            Block(5, 32, 96, 64, hswish(), SeModule(64), (2,1)),   
            #torch.Size([2, 64, 4, 140])
            Block(5, 64, 240, 64, hswish(), SeModule(64), 1),
            #torch.Size([2, 40, 4, 140])
            Block(5, 64, 240, 64, hswish(), SeModule(64), 1),
            #torch.Size([2, 64, 4, 140])
            Block(5, 64, 120, 128, hswish(), SeModule(128), 1),
            #torch.Size([2, 128, 4, 140])
            Block(5, 128, 144, 128, hswish(), SeModule(128), 1),
            #torch.Size([2, 128, 4, 140])
            Block(5, 128, 288, 256, hswish(), SeModule(256), (2,1)),
            #orch.Size([2, 256, 2, 140])
            Block(5, 256, 576, 256, hswish(), SeModule(256), 1),
            #orch.Size([2, 256, 2, 140])
            Block(5, 256, 576, 256, hswish(), SeModule(256), 1),
            #orch.Size([2, 256, 2, 140])
        )
        
        self.conv2=nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False), #[1, 256, 2, 140]  
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),    #[1, 256, 1, 70]  
        )
        
        self.conv3 = nn.Conv2d(256, 512, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(512)
        self.hs3 = hswish()
        self.init_params()
        
        self.rnn = nn.Sequential(
        BidirectionalLSTM(512, nh, nh),
        BidirectionalLSTM(nh, nh, nclass)
        )
        
        
    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
        
    def forward(self, x):
        #特征提取网络
        out = self.hs1(self.bn1(self.conv1(x)))
        out=self.blok(out)
        out=self.conv2(out)
        out = self.hs3(self.bn3(self.conv3(out)))
        #序列RNN
        b, c, h, w = out.size() #output: ([10, 512, 1, 251])
        assert h == 1, '图片高度经过卷积之后必须为1'
        out = out.squeeze(2)   #output: ([10, 512, 251])
        out = out.permute(2, 0, 1)  # [w, b, c]
        #print(conv.shape) #([251, 10, 11])
        out = self.rnn(out)     # seq * batch * n_classes// 25 × batchsize × 251（隐藏节点个数）
        return out


    
# net1=DRNet()
# net3=MobileNetV3_Small()
#summary(net1,(8, 32, 280),batch_size=1,device="cpu")
#summary(net3,(8, 32, 280),batch_size=1,device="cpu")
# summary(net1,(1, 32, 280),batch_size=1,device="cpu")

