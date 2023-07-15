# # Code copied from https://github.com/meliketoy/wide-resnet.pytorch

# import torch
# import torch.nn as nn
# import torch.nn.init as init
# import torch.nn.functional as F
# from torch.autograd import Variable

# import numpy as np

# def conv3x3(in_planes, out_planes, stride=1):
#     return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=True)

# def conv_init(m):
#     classname = m.__class__.__name__
#     if classname.find('Conv') != -1:
#         init.xavier_uniform_(m.weight, gain=np.sqrt(2))
#         init.constant_(m.bias, 0)
#     elif classname.find('BatchNorm') != -1:
#         init.constant_(m.weight, 1)
#         init.constant_(m.bias, 0)

# class wide_basic(nn.Module):
#     def __init__(self, in_planes, planes, dropout_rate, stride=1):
#         super(wide_basic, self).__init__()
#         self.bn1 = nn.BatchNorm2d(in_planes)
#         self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, padding=1, bias=True)
#         self.dropout = nn.Dropout(p=dropout_rate)
#         self.bn2 = nn.BatchNorm2d(planes)
#         self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=True)

#         self.shortcut = nn.Sequential()
#         if stride != 1 or in_planes != planes:
#             self.shortcut = nn.Sequential(
#                 nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=True),
#             )

#     def forward(self, x):
#         out = self.dropout(self.conv1(F.relu(self.bn1(x))))
#         out = self.conv2(F.relu(self.bn2(out)))
#         out += self.shortcut(x)

#         return out

# class Wide_ResNet(nn.Module):
#     def __init__(self, depth, widen_factor, dropout_rate, num_classes):
#         super(Wide_ResNet, self).__init__()
#         self.in_planes = 16

#         assert ((depth-4)%6 ==0), 'Wide-resnet depth should be 6n+4'
#         n = (depth-4)/6
#         k = widen_factor

#         print('| Wide-Resnet %dx%d' %(depth, k))
#         nStages = [16, 16*k, 32*k, 64*k]

#         self.conv1 = conv3x3(3,nStages[0])
#         self.layer1 = self._wide_layer(wide_basic, nStages[1], n, dropout_rate, stride=1)
#         self.layer2 = self._wide_layer(wide_basic, nStages[2], n, dropout_rate, stride=2)
#         self.layer3 = self._wide_layer(wide_basic, nStages[3], n, dropout_rate, stride=2)
#         self.bn1 = nn.BatchNorm2d(nStages[3], momentum=0.9)
#         self.linear = nn.Linear(nStages[3], num_classes)

#     def _wide_layer(self, block, planes, num_blocks, dropout_rate, stride):
#         strides = [stride] + [1]*(int(num_blocks)-1)
#         layers = []

#         for stride in strides:
#             layers.append(block(self.in_planes, planes, dropout_rate, stride))
#             self.in_planes = planes

#         return nn.Sequential(*layers)

#     def forward(self, x):
#         out = self.conv1(x)
#         out = self.layer1(out)
#         out = self.layer2(out)
#         out = self.layer3(out)
#         out = F.relu(self.bn1(out))
#         out = F.avg_pool2d(out, 8)
#         out = out.view(out.size(0), -1)
#         out = self.linear(out)

#         return out

# if __name__ == '__main__':
#     net=Wide_ResNet(28, 10, 0.3, 10)
#     y = net(Variable(torch.randn(1,3,32,32)))

#     print(y.size())

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def mish(x):
    """Mish: A Self Regularized Non-Monotonic Neural Activation Function (https://arxiv.org/abs/1908.08681)"""
    return x * torch.tanh(F.softplus(x))


class PSBatchNorm2d(nn.BatchNorm2d):
    """How Does BN Increase Collapsed Neural Network Filters? (https://arxiv.org/abs/2001.11216)"""

    def __init__(self, num_features, alpha=0.1, eps=1e-05, momentum=0.001, affine=True, track_running_stats=True):
        super().__init__(num_features, eps, momentum, affine, track_running_stats)
        self.alpha = alpha

    def forward(self, x):
        return super().forward(x) + self.alpha


class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, drop_rate=0.0, activate_before_residual=False):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes, momentum=0.001)
        self.relu1 = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes, momentum=0.001)
        self.relu2 = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.drop_rate = drop_rate
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                                                                padding=0, bias=False) or None
        self.activate_before_residual = activate_before_residual

    def forward(self, x):
        if not self.equalInOut and self.activate_before_residual == True:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))
        if self.drop_rate > 0:
            out = F.dropout(out, p=self.drop_rate, training=self.training)
        out = self.conv2(out)
        return torch.add(x if self.equalInOut else self.convShortcut(x), out)


class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, drop_rate=0.0, activate_before_residual=False):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(
            block, in_planes, out_planes, nb_layers, stride, drop_rate, activate_before_residual)

    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, drop_rate, activate_before_residual):
        layers = []
        for i in range(int(nb_layers)):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes,
                                i == 0 and stride or 1, drop_rate, activate_before_residual))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)


class Wide_ResNet(nn.Module):
    def __init__(self, depth, widen_factor, drop_rate, num_classes):
        super(Wide_ResNet, self).__init__()
        channels = [16, 16*widen_factor, 32*widen_factor, 64*widen_factor]
        assert((depth - 4) % 6 == 0)
        n = (depth - 4) / 6
        block = BasicBlock
        # 1st conv before any network block
        self.conv1 = nn.Conv2d(3, channels[0], kernel_size=3, stride=1,
                               padding=1, bias=False)
        # 1st block
        self.block1 = NetworkBlock(
            n, channels[0], channels[1], block, 1, drop_rate, activate_before_residual=True)
        # 2nd block
        self.block2 = NetworkBlock(
            n, channels[1], channels[2], block, 2, drop_rate)
        # 3rd block
        self.block3 = NetworkBlock(
            n, channels[2], channels[3], block, 2, drop_rate)
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(channels[3], momentum=0.001)
        self.relu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.fc = nn.Linear(channels[3], num_classes)
        self.channels = channels[3]

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = F.adaptive_avg_pool2d(out, 1)
        out = out.view(-1, self.channels)
        return self.fc(out)