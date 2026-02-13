import math
import torch
import functools
import numpy as np
import torch.nn as nn
from typing import List
from torch import Tensor
import torch.nn.functional as F
from collections import OrderedDict
from typing import Iterable, Optional

#------------------------------------------
#            ECAPA-TDNN code
#------------------------------------------

class Res2Conv1dReluBn(nn.Module):
    """
    in_channels == out_channels == channels
    """

    def __init__(self,
                 channels,
                 kernel_size=1,
                 stride=1,
                 padding=0,
                 dilation=1,
                 bias=True,
                 scale=4):
        super().__init__()
        assert channels % scale == 0, "{} % {} != 0".format(channels, scale)
        self.scale = scale
        self.width = channels // scale
        self.nums = scale if scale == 1 else scale - 1

        self.convs = []
        self.bns = []
        for i in range(self.nums):
            self.convs.append(
                nn.Conv1d(self.width,
                          self.width,
                          kernel_size,
                          stride,
                          padding,
                          dilation,
                          bias=bias))
            self.bns.append(nn.BatchNorm1d(self.width))
        self.convs = nn.ModuleList(self.convs)
        self.bns = nn.ModuleList(self.bns)

    def forward(self, x):
        out = []
        spx = torch.split(x, self.width, 1)
        sp = spx[0]
        for i, (conv, bn) in enumerate(zip(self.convs, self.bns)):
            # Order: conv -> relu -> bn
            if i >= 1:
                sp = sp + spx[i]
            sp = conv(sp)
            sp = bn(F.relu(sp))
            out.append(sp)
        if self.scale != 1:
            out.append(spx[self.nums])
        out = torch.cat(out, dim=1)

        return out


''' Conv1d + BatchNorm1d + ReLU
'''


class Conv1dReluBn(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=1,
                 stride=1,
                 padding=0,
                 dilation=1,
                 bias=True):
        super().__init__()
        self.conv = nn.Conv1d(in_channels,
                              out_channels,
                              kernel_size,
                              stride,
                              padding,
                              dilation,
                              bias=bias)
        self.bn = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        return self.bn(F.relu(self.conv(x)))


''' The SE connection of 1D case.
'''


class SE_Connect(nn.Module):

    def __init__(self, channels, se_bottleneck_dim=128):
        super().__init__()
        self.linear1 = nn.Linear(channels, se_bottleneck_dim)
        self.linear2 = nn.Linear(se_bottleneck_dim, channels)

    def forward(self, x):
        out = x.mean(dim=2)
        out = F.relu(self.linear1(out))
        out = torch.sigmoid(self.linear2(out))
        out = x * out.unsqueeze(2)

        return out


''' SE-Res2Block of the ECAPA-TDNN architecture.
'''


class SE_Res2Block(nn.Module):

    def __init__(self, channels, kernel_size, stride, padding, dilation,
                 scale):
        super().__init__()
        self.se_res2block = nn.Sequential(
            Conv1dReluBn(channels,
                         channels,
                         kernel_size=1,
                         stride=1,
                         padding=0),
            Res2Conv1dReluBn(channels,
                             kernel_size,
                             stride,
                             padding,
                             dilation,
                             scale=scale),
            Conv1dReluBn(channels,
                         channels,
                         kernel_size=1,
                         stride=1,
                         padding=0), SE_Connect(channels))

    def forward(self, x):
        return x + self.se_res2block(x)

def get_same_padding(kernel_size, dilation):
    k, d = kernel_size, dilation
    total_padding = d * (k - 1)
    left_pad = total_padding // 2
    right_pad = total_padding - left_pad
    assert left_pad==right_pad
    return left_pad


class ECAPA_TDNN(nn.Module):
    def __init__(self,
                 in_channels=128,
                 out_channels=128,
                 scale=8,
                 blocks_setup=[
                     (3,2),
                     (3,3),
                     # (3,5)
                 ],
                ):
        super().__init__()

        hid_dim = out_channels
        self.stem = Conv1dReluBn(in_channels,
                                   hid_dim,
                                   kernel_size=5,
                                   padding=2)

        blocks = []
        self.num_blocks = len(blocks_setup)
        for block_ind, (ker_size, dilation) in enumerate(blocks_setup):
            block = SE_Res2Block(hid_dim, kernel_size=ker_size, stride=1, 
                                 padding=get_same_padding(ker_size, dilation),
                                 dilation=dilation, scale=scale)
            setattr(self,f"block{block_ind}",block)

        cat_channels = hid_dim * self.num_blocks
        self.conv = nn.Conv1d(cat_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.stem(x)
        outs = []
        for block_ind in range(self.num_blocks):
            block = getattr(self,f"block{block_ind}")
            x = block(x)
            outs.append(x)

        out = torch.cat(outs, dim=1)
        out = self.conv(out)
        return out
