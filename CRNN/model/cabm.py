import torch
import torch.nn as nn


# 通道注意力模块
class ChannelAttentionModule(nn.Module):
    def __init__(self, channel=512, reduction=1):
        super(ChannelAttentionModule, self).__init__()
        mid_channel = channel #/ reduction
        # 使用自适应池化缩减map的大小，保持通道不变
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.shared_MLP = nn.Sequential(
            nn.Linear(in_features=channel, out_features=mid_channel),
            nn.ReLU(),
            nn.Linear(in_features=mid_channel, out_features=channel)
        )
        self.sigmoid = nn.Sigmoid()
        # self.act=SiLU()

    def forward(self, x):
        avgout = self.shared_MLP(self.avg_pool(x).view(x.size(0), -1)).unsqueeze(2).unsqueeze(3)
        # print('avg',avgout.size())  # [16, 256, 1, 1]
        maxout = self.shared_MLP(self.max_pool(x).view(x.size(0), -1)).unsqueeze(2).unsqueeze(3)
        # print('max',maxout.size())  # [16, 256, 1, 1]
        return self.sigmoid(avgout + maxout)


# 空间注意力模块
class SpatialAttentionModule(nn.Module):
    def __init__(self):
        super(SpatialAttentionModule, self).__init__()
        self.conv2d = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=2, stride=1, padding=0)
        # self.act=SiLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # map尺寸不变，缩减通道
        avgout = torch.mean(x, dim=1, keepdim=True)
        # print('savg',avgout.size())  # [16, 1, 384, 384]
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        # print('smax',maxout.size())  # [16, 1, 384, 384]
        out = torch.cat([avgout, maxout], dim=1)
        # print('s+s',out.size())  # [16, 2, 384, 384]
        out = self.sigmoid(self.conv2d(out))
        # print('ssigmoid',out.size())  # [16, 1, 384, 384]
        return out


# CBAM模块
class CBAM(nn.Module):
    def __init__(self, in_channel=512, out_channels=512):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttentionModule()
        self.spatial_attention = SpatialAttentionModule()
        self.out_channel = out_channels
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(in_channel, out_channels, kernel_size=2, stride=1, padding=0)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        residual = x
        # print('x', x.size())
        out = self.channel_attention(x) * x
        out = self.spatial_attention(out) * out
        out = residual + out
        # print(out.size())
        out = self.relu(out)
        out = self.conv(out)
        out = self.bn(out)
        out = self.relu(out)
        return out

if __name__ == '__main__':
    model = CBAM()
    input_tensor = torch.randn(512, 512, 192, 192)

    prediction1 = model(input_tensor)
    print(prediction1)