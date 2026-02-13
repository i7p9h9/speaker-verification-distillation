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
from torch.nn.modules.utils import _pair

#------------------------------------------
#       ResNet basice block (+fwSE)
#------------------------------------------

class CausalPaddingNd(nn.Module):
    def __init__(self, dim=1, kernel_size=3, dilation=1, mode='reflect'):
        """
        Args:
            dim (int): The number of dimensions (1 or 2).
            kernel_size: For dim==1, an int; for dim==2, an int or tuple (height, width).
            dilation: For dim==1, an int; for dim==2, an int or tuple (dilation_height, dilation_width).
            
        The padding is computed only for the causal (i.e. last) dimension.
        """
        super().__init__()
        if dim not in (1, 2):
            raise ValueError("dim must be 1 or 2")
        self.dim = dim

        self.mode = mode
        self.pad_t = 0
        self.pad_b = 0
        self.pad_w_r = 0
        if self.dim == 1:
            # For 1D case, expect ints.
            if not isinstance(kernel_size, int):
                raise ValueError("For dim=1, kernel_size must be an int")
            if not isinstance(dilation, int):
                raise ValueError("For dim=1, dilation must be an int")
            self.kernel_size = kernel_size
            self.dilation = dilation
            # Compute causal left padding for 1D.
            self.pad_w_l = (self.kernel_size - 1) * self.dilation
        else:
            # For 2D, use _pair to ensure tuple conversion.
            self.kernel_size = _pair(kernel_size)
            self.dilation = _pair(dilation)
            # For height axis (H): compute symmetric same padding.
            pad_h_total = (self.kernel_size[0] - 1) * self.dilation[0]
            self.pad_t = pad_h_total // 2
            self.pad_b = pad_h_total - self.pad_t

            # For width axis (W): causal padding (only pad the left side).
            self.pad_w_l = (self.kernel_size[1] - 1) * self.dilation[1]
            self.pad_w_r = 0

    def extra_repr(self) -> str:
        return f"pt={self.pad_t},pb={self.pad_b},pl={self.pad_w_l},pw={self.pad_w_r}"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.dim == 1:
            # Input shape: (batch, channels, length)
            # F.pad expects (pad_left, pad_right)
            return F.pad(x, (self.pad_w_l, 0), mode=self.mode)
        else:
            # Input shape: (batch, channels, height, width)
            # F.pad for 2D expects (pad_left, pad_right, pad_top, pad_bottom)
            return F.pad(x, (self.pad_w_l, self.pad_w_r, self.pad_t, self.pad_b), mode=self.mode)
            
class fwSEBlock(nn.Module):
    """
    Squeeze-and-Excitation block
    link: https://arxiv.org/pdf/1709.01507.pdf
    PyTorch implementation
    """
    def __init__(self, num_freq, num_feats=64):
        super(fwSEBlock, self).__init__()
        self.squeeze = nn.Linear(num_freq, num_feats)
        self.exitation = nn.Linear(num_feats, num_freq)

        self.activation = nn.ReLU()  # Assuming ReLU, modify as needed

    def forward(self, inputs):
        # [bs, C, F, T]
        x = torch.mean(inputs, dim=[1, 3])
        x = self.squeeze(x)
        x = self.activation(x)
        x = self.exitation(x)
        x = torch.sigmoid(x)
        # Reshape and apply excitation
        x = x[:,None,:,None]
        x = inputs * x
        return x

class ResBasicBlock(nn.Module):
    def __init__(self, inc, outc, num_freq, stride=1, se_channels=64, Gdiv=4, use_fwSE=False, causal=False):
        super().__init__()
        # if inc//Gdiv==0:
        #     Gdiv = inc
        if not causal:
            self.conv1 = nn.Conv2d(inc, inc if Gdiv is not None else outc, kernel_size=3, stride=stride, padding=1, bias=False, 
                                   groups=inc//Gdiv if Gdiv is not None else 1)
        else:
            self.conv1 = nn.Sequential(
                CausalPaddingNd(dim=2, kernel_size=3, dilation=1),
                nn.Conv2d(inc, inc if Gdiv is not None else outc, kernel_size=3, 
                          stride=stride, padding=0, bias=False, 
                          groups=inc//Gdiv if Gdiv is not None else 1)
                
            )
            
        if Gdiv is not None:
            self.conv1pw = nn.Conv2d(inc, outc, 1)
        else:
            self.conv1pw = nn.Identity()
            
        self.bn1 = nn.BatchNorm2d(outc)
        if not causal:
            self.conv2 = nn.Conv2d(outc, outc, kernel_size=3, padding=1, bias=False, groups=outc//Gdiv if Gdiv is not None else 1)
        else:
            self.conv2 = nn.Sequential(
                CausalPaddingNd(dim=2, kernel_size=3, dilation=1),
                nn.Conv2d(outc, outc, kernel_size=3, padding=0, bias=False, 
                          groups=outc//Gdiv if Gdiv is not None else 1)
            )
        
        if Gdiv is not None:
            self.conv2pw = nn.Conv2d(outc, outc, 1)
        else:
            self.conv2pw = nn.Identity()
            
        self.bn2 = nn.BatchNorm2d(outc)
        self.relu = nn.ReLU(inplace=True)
        
        if use_fwSE:
            self.se = fwSEBlock(num_freq, se_channels)
        else:
            self.se = nn.Identity()
            
        if outc != inc:
            self.downsample = nn.Sequential(
                nn.Conv2d(inc, outc, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(outc),
            )
        else:
            self.downsample = nn.Identity()
        
    def forward(self, x):
        residual = x

        out = self.conv1pw(self.conv1(x))
        out = self.relu(out)
        out = self.bn1(out)

        out = self.conv2pw(self.conv2(out))
        out = self.bn2(out)
        out = self.se(out)

        out += self.downsample(residual)
        out = self.relu(out)
        # print(out.size())
        return out
