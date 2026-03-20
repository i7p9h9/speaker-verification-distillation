import torch.nn as nn
import torch.nn.functional as F

class GRU(nn.Module):
    def __init__(self,*args,**kwargs):
        super(GRU, self).__init__()
        self.gru = nn.GRU(*args,**kwargs)

    def forward(self, x):
        # x : (bs,C,T) 
        return self.gru(x.permute((0,2,1)))[0].permute((0,2,1))
