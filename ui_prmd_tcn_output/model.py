import torch
from torch import nn
import torch.nn.functional as F

class TemporalBlock(nn.Module):
    def __init__(self, cin, cout, dilation, kernel, dropout):
        super().__init__()
        self.pad = (kernel-1)*dilation
        self.c1 = nn.Conv1d(cin,cout,kernel,dilation=dilation)
        self.c2 = nn.Conv1d(cout,cout,kernel,dilation=dilation)
        self.n1 = nn.LayerNorm(cout)
        self.n2 = nn.LayerNorm(cout)
        self.drop = nn.Dropout(dropout)
        self.skip = nn.Conv1d(cin,cout,1) if cin!=cout else nn.Identity()
    def forward(self,x):
        z = self.c1(F.pad(x,(self.pad,0)))
        z = self.drop(F.relu(self.n1(z.transpose(1,2)).transpose(1,2)))
        z = self.c2(F.pad(z,(self.pad,0)))
        z = self.drop(F.relu(self.n2(z.transpose(1,2)).transpose(1,2)))
        return F.relu(z+self.skip(x))

class PoseTCN(nn.Module):
    def __init__(self,cfg,mean,std):
        super().__init__()
        self.register_buffer('mean',torch.as_tensor(mean,dtype=torch.float32))
        self.register_buffer('std',torch.as_tensor(std,dtype=torch.float32))
        blocks=[]
        for i,d in enumerate(cfg['dilations']):
            blocks.append(TemporalBlock(36 if i==0 else cfg['channels'],cfg['channels'],
                                        d,cfg['kernel'],cfg['dropout']))
        self.blocks=nn.Sequential(*blocks)
        self.head=nn.Linear(cfg['channels'],5)
    def forward(self,x):
        x=((x-self.mean)/self.std).transpose(1,2)
        return self.head(self.blocks(x).mean(dim=-1))
