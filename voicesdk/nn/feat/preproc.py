import torch
import torch.nn.functional as F
from torch import nn


class NormalizeAudio(nn.Module):
    def __init__(self, eps:float=1e-6, squeeze:bool=False):
        super().__init__()
        self.eps = eps
        self.squeeze = squeeze

    def forward(self,x):
        if x.ndim == 2:
            x = x.unsqueeze(1)
        x = (x - x.mean(dim=2, keepdim=True)) /\
                     (x.std(dim=2, keepdim=True, unbiased=False) + self.eps)
        if self.squeeze:
            x = x.squeeze(1)
        return x


class PreEmphasis(torch.nn.Module):
    def __init__(self, coef: float = 0.97):
        super().__init__()
        self.coef = coef
        self.register_buffer(
            'flipped_filter', torch.FloatTensor([-self.coef, 1.]).unsqueeze(0).unsqueeze(0)
        )

    def forward(self, x):
        if x.ndim == 2:
            x = x.unsqueeze(1)
        x = F.pad(x, (1, 0), 'reflect')
        return F.conv1d(x, self.flipped_filter).squeeze(1)  # pylint: disable=not-callable
