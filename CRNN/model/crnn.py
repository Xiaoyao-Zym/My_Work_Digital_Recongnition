import torch
import torch.nn as nn
#from model.cabm import CBAM
import torch.nn.functional as F
from torch.autograd import Variable
class BidirectionalLSTM(nn.Module):
    def __init__(self, nIn, nHidden, nOut):
        """
        :param nIn: 输入层神经元个数
        :param nHidden: 隐藏层神经元个数
        :param nOut: 输出层神经元个数
        """
        super(BidirectionalLSTM, self).__init__()
        # 双向LSTM
        self.rnn = nn.LSTM(nIn, nHidden, bidirectional=True)
        # 两个方向的隐藏层单元频在一起，所以nHidden*2
        self.embedding = nn.Linear(nHidden * 2, nOut)

    def forward(self, input):
        recurrent, _ = self.rnn(input)
        # T:时间序列 b:batch_size h:隐藏层神经元
        T, b, h = recurrent.size()
        t_rec = recurrent.view(T * b, h)
        output = self.embedding(t_rec)
        output = output.view(T, b, -1)
        return output

class AttentionCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(AttentionCell, self).__init__()
        self.i2h = nn.Linear(input_size, hidden_size,bias=False)
        self.h2h = nn.Linear(hidden_size, hidden_size)
        self.score = nn.Linear(hidden_size, 1, bias=False)
        self.rnn = nn.GRUCell(input_size, hidden_size)
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.processed_batches = 0

    def forward(self, prev_hidden, feats):
        self.processed_batches = self.processed_batches + 1
        nC = feats.size(0)
        nB = feats.size(1)
        nT = feats.size(2)
        hidden_size = self.hidden_size
        input_size = self.input_size

        feats_proj = self.i2h(feats.view(-1,nC))
        prev_hidden_proj = self.h2h(prev_hidden).view(1,nB, hidden_size).expand(nT, nB, hidden_size).contiguous().view(-1, hidden_size)
        emition = self.score(torch.tanh(feats_proj + prev_hidden_proj).view(-1, hidden_size)).view(nT,nB).transpose(0,1)
        alpha = F.softmax(emition, dim=1) # nB * nT

        if self.processed_batches % 10000 == 0:
            print('emition ', list(emition.data[0]))
            print('alpha ', list(alpha.data[0]))

        feats=feats.transpose(0, 2)
        context = (feats * alpha.transpose(0,1).contiguous().view(nT,nB,1).expand(nT, nB, nC)).sum(0).squeeze(0)
        cur_hidden = self.rnn(context, prev_hidden)
        return cur_hidden, alpha

class Attention(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(Attention, self).__init__()
        self.attention_cell = AttentionCell(input_size, hidden_size)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.generator = nn.Linear(hidden_size, num_classes-1)
        self.processed_batches = 0

    def forward(self, feats, text_length):
        self.processed_batches = self.processed_batches + 1
        nC = feats.size(0)
        nB = feats.size(1)
        nT = feats.size(2)
        hidden_size = self.hidden_size
        input_size = self.input_size
        print("in=", input_size)
        assert(input_size == nC)
        assert(nB == text_length.numel())

        num_steps = text_length.data.max()
        num_labels = text_length.data.sum()

        output_hiddens = Variable(torch.zeros(num_steps, nB, hidden_size).type_as(feats.data))
        hidden = Variable(torch.zeros(nB,hidden_size).type_as(feats.data))
        max_locs = torch.zeros(num_steps, nB)
        max_vals = torch.zeros(num_steps, nB)
        for i in range(num_steps):
            hidden, alpha = self.attention_cell(hidden, feats)
            output_hiddens[i] = hidden
            if self.processed_batches % 500 == 0:
                max_val, max_loc = alpha.data.max(1)
                max_locs[i] = max_loc.cpu()
                max_vals[i] = max_val.cpu()
        if self.processed_batches % 500 == 0:
            print('max_locs', list(max_locs[0:text_length.data[0],0]))
            print('max_vals', list(max_vals[0:text_length.data[0],0]))
        new_hiddens = Variable(torch.zeros(num_labels, hidden_size).type_as(feats.data))
        b = 0
        start = 0
        for length in text_length.data:
            new_hiddens[start:start+length] = output_hiddens[0:length,b,:]
            start = start + length
            b = b + 1
        probs = self.generator(new_hiddens)
        return probs

class CRNN(nn.Module):

    def __init__(self, imgH, nc, nclass, nh):
        """

        :param imgH: 图片高度
        :param nc: 图片通道数
        :param nclass: 类别个数
        :param nh: RNN中隐藏层神经元个数
        """
        super(CRNN, self).__init__()
        assert imgH % 16 == 0, '图片高度必须是16的倍数，建议32'

        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=nc, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 1), padding=(0, 1)),

            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),

            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 1), padding=(0, 1)),

            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=2, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
        )
        # self.cabm=nn.Sequential(
        #     CBAM()
        # )

        self.rnn = nn.Sequential(
            BidirectionalLSTM(512, nh, nh),
            BidirectionalLSTM(nh, nh, nclass)
        )
        self.attention = Attention(nh, nh, nclass)

    def forward(self, input,  length):
        conv = self.cnn(input) #input: [10, 1, 32, 280], 
        print(conv.size())
        b, c, h, w = conv.size() #output: ([10, 512, 1, 251])
        assert h == 1, '图片高度经过卷积之后必须为1'
        conv = conv.squeeze(2)   #output: ([10, 512, 251])
        conv = conv.permute(2, 0, 1)  # [w, b, c]
        print(conv.shape) #([251, 10, 11])
        output = self.rnn(conv)     # seq * batch * n_classes// 25 × batchsize × 251（隐藏节点个数）
        print(output.shape)     #([251, 10, 11])
        print(length)
        #output = self.attention(output, length)
        print(output.shape)
        return output
