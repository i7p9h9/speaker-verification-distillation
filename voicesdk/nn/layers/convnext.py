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
#           ConvNeXtV2 block
#------------------------------------------

class CausalPaddingNd(nn.Module):
    def __init__(self, dim=1, kernel_size=3, dilation=1, mode='constant'):
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

MaxPoolNd = {
    1 : nn.MaxPool1d,
    2 : nn.MaxPool2d
}

ConvNd = {
    1 : nn.Conv1d,
    2 : nn.Conv2d
}

BatchNormNd = {
    1 : nn.BatchNorm1d,
    2 : nn.BatchNorm2d
}

# https://github.com/facebookresearch/ConvNeXt/blob/main/models/convnext.py
# https://github.com/facebookresearch/ConvNeXt-V2/blob/main/models/convnextv2.py
class ConvNeXtLikeBlock(nn.Module):
    def __init__(self, C, dim=2, kernel_sizes=[(3,3),], Gdiv=1, padding='same', activation='gelu', glu=False, causal=False):
        super().__init__()
        # if C//Gdiv==0:
        #     Gdiv = C
        _kernel_sizes = []
        _dilations = []
        if dim == 2:
            for ks in kernel_sizes:
                if isinstance(ks[0],int):
                    _kernel_sizes.append(ks)
                    _dilations.append((1,1))
                elif isinstance(ks[0],(tuple,list)) and isinstance(ks[1],(tuple,list)):
                    _kernel_sizes.append(ks[0])
                    _dilations.append(ks[1])
                else:
                    assert False
        else:
            for ks in kernel_sizes:
                if isinstance(ks,int):
                    _kernel_sizes.append(ks)
                    _dilations.append(1)
                elif isinstance(ks,(tuple,list)):
                    _kernel_sizes.append(ks[0])
                    _dilations.append(ks[1])
                else:
                    assert False

        if not causal:
            self.dwconvs = nn.ModuleList(modules=[
                ConvNd[dim](C, C, kernel_size=ks, dilation=dil,
                                    padding=padding, groups=C//Gdiv if Gdiv is not None else 1) 
                for ks,dil in zip(_kernel_sizes,_dilations)
            ])
        else:
            modules = []
            for ks,dil in zip(_kernel_sizes,_dilations):
                modules.append(nn.Sequential(
                    CausalPaddingNd(dim=dim,kernel_size=ks, dilation=dil),
                    ConvNd[dim](C, C, kernel_size=ks, dilation=dil,
                                    padding=0, groups=C//Gdiv if Gdiv is not None else 1)
                ))
            self.dwconvs = nn.ModuleList(modules=modules)
        self.norm = BatchNormNd[dim](C * len(kernel_sizes))
        if activation == 'gelu':
            self.act = nn.GELU()
        elif activation == 'relu':
            self.act = nn.ReLU()
        self.glu = glu
        pwconv_out_chn = C
        if self.glu:
            pwconv_out_chn = C * 2
        self.pwconv1 = ConvNd[dim](C * len(kernel_sizes), pwconv_out_chn, 1) # pointwise/1x1 convs, implemented with linear layers


    def forward(self, x):
        skip = x
        x = torch.cat([dwconv(x) for dwconv in self.dwconvs],dim=1)
        x = self.act(self.norm(x))
        x = self.pwconv1(x)
        if self.glu:
            x = F.glu(x, dim=1)
        x = skip + x
        return x