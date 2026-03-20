import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm(nn.Module): # ⚡
    """ LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, T, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, T).
    """
    def __init__(self, C, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(C))
        self.bias = nn.Parameter(torch.zeros(C))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.C = (C, )

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.C, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)

            w = self.weight
            b = self.bias
            for _ in range(x.ndim-2):
                w = w.unsqueeze(-1)
                b = b.unsqueeze(-1)
            x = w * x + b # ⚡
            return x

    def extra_repr(self) -> str:
        return ", ".join([f"{k}={v}" for k,v in {"C" : self.C, "data_format" : self.data_format, "eps" : self.eps}.items()])
