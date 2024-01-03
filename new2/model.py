import torch
import torch.nn as nn
from torchvision import models
import torchvision.models.resnet as resnet
from torchvision.models.convnext import convnext_base as convnext

class SBConvNext(nn.Module):
    '''
    Convnext
    '''

    def __init__(self):
        super(SBConvNext, self).__init__()
        self.cnn = convnext(weights='ConvNeXt_Base_Weights.DEFAULT')  # 不知道输入图像的size应该设置为多少
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(1000, 128),  # ConvNeXt输出默认大小为1000
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(128, 3),
            # torch.nn.Sigmoid()  # 原来的方法有，在此删除
        )

    def forward(self, x):
        return self.mlp(self.cnn(x))

class SBResNet(torch.nn.Module):
    '''
    Resnet
    '''

    def __init__(self):
        super(SBResNet, self).__init__()

        self.cnn = resnet.resnet101(weights='ResNet101_Weights.DEFAULT')
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(self.cnn.fc.weight.shape[0], 128),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(128, 3),
        )

    def forward(self, x):
        return self.mlp(self.cnn(x))