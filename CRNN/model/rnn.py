import torch.nn as nn
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