def to1d(x):
    size = x.size()
    bs,c,f,t = size
    x = x.permute((0,2,1,3)) # bs, f, c, t
    x = x.reshape((bs,c*f,t)) # bs, c*f, t
    return x 

def to2d(x,c,f):
    bs,cf,t = x.size()
    out = x.reshape((bs, f, c, t)) # bs,f,c,t
    out = out.permute((0,2,3,1)) # bs,c,t,f
    return out

class to2d(nn.Module):
    def __init__(self, f, c):
        super().__init__()
        self.f = f
        self.c = c

    def forward(self,x):
        size = x.size()
        # bs,cf,t = tuple(size)
        bs,cf,t = size
        out = x.reshape((bs,self.f,self.c,t)).permute((0,2,1,3))
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
    def __init__(self, N, C, 
                 requires_grad=True):
        super().__init__()
        self.w = nn.Parameter(torch.zeros(1,N,C,1),
                              requires_grad=requires_grad)

    def forward(self, xs):
        xs = torch.cat([t.unsqueeze(1) for t in xs],dim=1)
        w = F.softmax(self.w,dim=1)
        x = (w*xs).sum(dim=1)
        # print(f"weigth1d : {x.size()}")
        return x

    def extra_repr(self) -> str:
        return f"w={tuple(self.w.size())}"