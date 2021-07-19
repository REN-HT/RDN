import torch
from torch import nn


# 网络默认为2倍上采样，可自己更改3或4倍上采样
class RDB_Conv(nn.Module):
    def __init__(self, inchannel, grow_rate):
        super(RDB_Conv, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(inchannel, grow_rate, 3, 1, 1),
                                  nn.ReLU(inplace=True))

    def forward(self, x):
        out = self.conv(x)
        out = torch.cat((x, out), 1)
        return out


class RDB(nn.Module):
    def __init__(self, inchannel, outchannel, grow_rate, num_conv):
        super(RDB, self).__init__()
        convs = []
        for c in range(num_conv):
            convs.append(RDB_Conv(inchannel+c*grow_rate, grow_rate))
        self.conv = nn.Sequential(*convs)
        # 局部特征融合 local features fusion
        self.LFF = nn.Conv2d(inchannel+num_conv*grow_rate, outchannel, 1, 1, 0)

    def forward(self, x):
        out = self.conv(x)
        out = self.LFF(out)
        return out + x


class RDNet(nn.Module):
    def __init__(self, inchannel=3, midchannel=64, outchannel=3, grow_rate=64, num_conv=8, num_block=20, up_scale=2):
        super(RDNet, self).__init__()
        self.up_scale = up_scale
        self.D = num_block
        # 浅层特征提取
        self.conv1 = nn.Conv2d(inchannel, midchannel, 3, 1, 1)
        self.conv2 = nn.Conv2d(midchannel, midchannel, 3, 1, 1)
        # RDB块
        self.RDBS = nn.ModuleList()
        for n in range(num_block):
            self.RDBS.append(RDB(midchannel, midchannel, grow_rate, num_conv))
        # 全局特征融合 global feature fusion
        self.GFF = nn.Sequential(nn.Conv2d(num_block*midchannel, midchannel, 1, 1, 0),
                                 nn.Conv2d(midchannel, midchannel, 3, 1, 1))
        # 上采样
        if self.up_scale == 2 or self.up_scale == 3:
            self.up_net = nn.Sequential(nn.Conv2d(midchannel, midchannel*self.up_scale*self.up_scale, 3, 1, 1),
                                        nn.PixelShuffle(self.up_scale))
        elif self.up_scale == 4:
            self.up_net = nn.Sequential(nn.Conv2d(midchannel, midchannel*self.up_scale, 3, 1, 1),
                                        nn.PixelShuffle(2),
                                        nn.Conv2d(midchannel, midchannel*self.up_scale, 3, 1, 1),
                                        nn.PixelShuffle(2))
        else:
            raise ValueError("up_scale must be 2 or 3 or 4.")

        self.last_conv = nn.Conv2d(midchannel, outchannel, 3, 1, 1)

        # # 网络参数初始化
        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         nn.init.kaiming_normal_(m.weight)
        #         if m.bias is not None:
        #             m.bias.data.zero_()

    def forward(self, x):
        f_1 = self.conv1(x)
        out = self.conv2(f_1)
        local_features = []
        for i in range(self.D):
            out = self.RDBS[i](out)
            local_features.append(out)
        out = self.GFF(torch.cat(local_features, 1))
        out = out + f_1
        out = self.up_net(out)
        out = self.last_conv(out)

        return out


# Charbonnier损失函数
class L1_Charbonnier_loss(nn.Module):
    """L1 Charbonnierloss."""
    def __init__(self):
        super(L1_Charbonnier_loss, self).__init__()
        self.eps = 1e-6

    def forward(self, X, Y):
        diff = torch.add(X, -Y)
        error = torch.sqrt(diff * diff + self.eps)
        loss = torch.sum(error)
        return loss
