#!/usr/bin/env python

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under: https://github.com/mpc001/Lipreading_using_Temporal_Convolutional_Networks/blob/master/LICENSE

# Ack: Code taken from Pingchuan Ma: https://github.com/mpc001/Lipreading_using_Temporal_Convolutional_Networks

import math
import torch.nn as nn
import pdb


#卷积3*3；使用bn层 不适用bias
def conv3x3(in_planes, out_planes, stride=1):                                 #卷积步长设为1
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)                                   #二维卷积方法，kernel_size卷积核大小，padding特征图填充宽度，bias偏置

#BN层的作用主要有三个:(1). 加快网络的训练和收敛的速度(2). 控制梯度爆炸防止梯度消失(3). 防止过拟合
#对基本块进行下采样；bn层 去除绝对差异 突出相对差异 适用于分类 步骤 求均值 方差 归一  调参 做归一化处理
def downsample_basic_block( inplanes, outplanes, stride ):
    return  nn.Sequential(                                                    #是一个Sequential网络，快速搭建网络
                nn.Conv2d(inplanes, outplanes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(outplanes),                                    #进行数据的归一化处理
            )
#二维
def downsample_basic_block_v2( inplanes, outplanes, stride ):
    return  nn.Sequential(
                # 二维平均池化操作，ceil_mode=True表示向上取整，count_include_pad表示计算均值的时候是否包含零填充
                nn.AvgPool2d(kernel_size=stride, stride=stride, ceil_mode=True, count_include_pad=False),
                nn.Conv2d(inplanes, outplanes, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(outplanes),
            )


#ResNetV2的网络深度有18，34，50，101，152。50层以下的网络基础块是BasicBlock
#残差结构
class BasicBlock(nn.Module):
    # expansion是对输出通道数的倍乘；resnet18中卷积核的个数没有发生变化 so =1
    expansion = 1

    # inplanes 输入特征矩阵深度 planes输出特征矩阵深度
    def __init__(self, inplanes, planes, stride=1, downsample=None, relu_type = 'relu' ):
        super(BasicBlock, self).__init__()
        # assert 表达式：（1）当表达式为真时，程序继续往下执行，只是判断，不做任何处理；（2）当表达式为假时，抛出AssertionError错误，并将 [参数] 输出
        assert relu_type in ['relu','prelu']                               #relu、prelu都是激活函数

        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        # relu激活函数 大于0为原值  小于0取0  inplace=true则不保留原值  一般设置为false
        # type of ReLU is an input option
        if relu_type == 'relu':
            self.relu1 = nn.ReLU(inplace=True)
            self.relu2 = nn.ReLU(inplace=True)
        elif relu_type == 'prelu':
            self.relu1 = nn.PReLU(num_parameters=planes)
            self.relu2 = nn.PReLU(num_parameters=planes)
        else:
            raise Exception('relu type not implemented')
        # --------

        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu2(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, relu_type = 'relu', gamma_zero = False, avg_pool_downsample = False):
        self.inplanes = 64
        self.relu_type = relu_type
        self.gamma_zero = gamma_zero
        self.downsample_block = downsample_basic_block_v2 if avg_pool_downsample else downsample_basic_block

        super(ResNet, self).__init__()
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        # default init
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
                #nn.init.ones_(m.weight)
                #nn.init.zeros_(m.bias)

        if self.gamma_zero:
            for m in self.modules():
                if isinstance(m, BasicBlock ):
                    m.bn2.weight.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):


        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = self.downsample_block( inplanes = self.inplanes, 
                                                 outplanes = planes * block.expansion, 
                                                 stride = stride )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, relu_type = self.relu_type))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, relu_type = self.relu_type))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x
