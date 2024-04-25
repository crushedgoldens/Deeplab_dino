import datetime
import os
from functools import partial

import numpy as np
import torch
from torch import nn
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim as optim
from torch.utils.data import DataLoader

import torch.nn.functional as F
import dino_feature_extracor

def batch_extractor(batch_size,train_dataset,layers):
    data=DataLoader(dataset=train_dataset,batch_size=batch_size)
    dino=dino_feature_extracor()
    feature=dino(data,layers)

    # 给每层命名，并存储到字典中
    layer_dict = {}
    for i in range(feature.shape[0]):
        layer_name = f'layer_{i}'  # 创建一个层的名称
        layer_dict[layer_name] = feature[i]

    return layer_dict

def feature_fusion1(deep,dino):
    # 将tensor2上采样到与tensor1相同的空间维度
    dino_upsampled = F.interpolate(dino, size=(128, 128), mode='bilinear', align_corners=False)

    # 将两个tensor在最后一个维度（通道维度）上拼接
    concatenated_tensor = torch.cat((deep, dino_upsampled), dim=2)

    # 定义一个1x1卷积层来减少通道数到24
    class ChannelReducer(nn.Module):
        def __init__(self, in_channels, out_channels):
            super(ChannelReducer, self).__init__()
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

        def forward(self, x):
            return self.conv(x)
        
    # 实例化1x1卷积层，输入通道数为tensor1和tensor2通道数之和，输出通道数为24
    channel_reducer = ChannelReducer(in_channels=deep.size(2) + dino_upsampled.size(2), out_channels=24)

    # 将拼接后的tensor传递给1x1卷积层
    output_tensor = channel_reducer(concatenated_tensor)

    # 输出tensor的形状应该是[batch_size, 24, H, W]，其中H和W是tensor1的空间维度
    print(output_tensor.shape)

def feature_fusion2(deep,dino):
    # 首先，我们需要将两个tensor调整到相同的空间维度以便于拼接
    # 这通常涉及到对较小的tensor进行上采样（upsampling）或对较大的tensor进行下采样（downsampling）
    # 这里我们选择将tensor2上采样到与tensor1相同的空间维度
    dino_upsampled = F.interpolate(dino, size=(128, 128), mode='bilinear', align_corners=False)

    # 然后，我们将两个tensor在最后一个维度（通道维度）上拼接
    # 拼接后的tensor形状将是[128, 128, 24+1024]
    concatenated_tensor = torch.cat((deep, dino_upsampled), dim=2)

    # 接下来，我们定义一个多层感知机（MLP）来将拼接后的tensor映射到通道数为24的tensor
    class MLP(nn.Module):
        def __init__(self):
            super(MLP, self).__init__()
            # 定义MLP的层
            self.fc1 = nn.Linear(128 * 128 * (24 + 1024), 128 * 128 * 24)  # 假设输入和输出的通道数是相同的

        def forward(self, x):
        # 展平tensor以适应全连接层
            x = x.view(x.size(0), -1)
            x = self.fc1(x)
        # 将输出重新塑形为原始的空间维度
            x = x.view(x.size(0), 24, 128, 128)
            return x

    # 实例化MLP
    mlp = MLP()

    # 将拼接后的tensor传递给MLP
    output_tensor = mlp(concatenated_tensor)

    # 输出tensor的形状应该是[128, 128, 24]
    print(output_tensor.shape)
