import torch
import torch.nn as nn
import torch.nn.functional as F

#------------------------------------------
#     ReDimNet main structural modules
#------------------------------------------

class to1d(nn.Module):
    def forward(self,x):
        size = x.size()
        # bs,c,f,t = tuple(size)
        bs,c,f,t = size
        return x.permute((0,2,1,3)).reshape((bs,c*f,t))

class to2d(nn.Module):
    def __init__(self, f, c):
        super().__init__()
        self.f = f
        self.c = c

    def forward(self,x):
        size = x.size()
        # bs,cf,t = tuple(size)
        bs,cf,t = size
        try:
            out = x.reshape((bs,self.f,self.c,t)).permute((0,2,1,3))
        except:
            print(f"ERROR : (C={self.c},F={self.f})")
            assert False
        # print(f"to2d : {out.size()}")
        return out

    def extra_repr(self) -> str:
        return f"f={self.f},c={self.c}"

class to1d_tfopt(nn.Module):
    def forward(self,x):
        bs,c,t,f = x.size() 
        return x.permute((0,3,1,2)).reshape((bs,c*f,t))

class to2d_tfopt(nn.Module):
    def __init__(self, f, c):
        super().__init__()
        self.f = f
        self.c = c

    def forward(self,x):
        bs,cf,t = x.size()
        out = x.reshape((bs, self.f, self.c, t)) # bs,f,c,t
        out = out.permute((0,2,3,1)) # bs,c,t,f
        return out

    def extra_repr(self) -> str:
        return f"f={self.f},c={self.c}"

class weigth1d(nn.Module):
    def __init__(self, N, C=None, sequential=False,
                 requires_grad=True):
        super().__init__()
        self.N = N
        self.sequential = sequential
        if C is None:
            C = 1
        self.w = nn.Parameter(torch.zeros(1,N,C,1),
                              requires_grad=requires_grad)

    def forward(self, xs):
        w = F.softmax(self.w,dim=1)
        if not self.sequential:
            xs = torch.cat([t.unsqueeze(1) for t in xs],dim=1)
            x = (w*xs).sum(dim=1)
            # print(f"weigth1d : {x.size()}")
        else:
            s = torch.zeros_like(xs[0])
            for i,t in enumerate(xs):
                s += t*w[:,i,:,:]
            x = s
            # x = sum([t*w[:,i,:,:] for i,t in enumerate(xs)])
        return x

    def extra_repr(self) -> str:
        return f"w={tuple(self.w.size())},sequential={self.sequential}"

class res1d(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        
    def forward(self, xs):
        assert len(xs) > 0
        # print([t.size() for t in xs])
        if len(xs) == 1:
            return xs[0]
        else:
            return xs[-1] + xs[-2]
