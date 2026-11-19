import torch
from torch import nn

class PoseLSTM(nn.Module):
    def __init__(self,cfg,mean,std,hidden=64,layers=2):
        super().__init__()
        self.register_buffer('mean',torch.as_tensor(mean,dtype=torch.float32))
        self.register_buffer('std',torch.as_tensor(std,dtype=torch.float32))
        self.lstm=nn.LSTM(cfg['input_features'],hidden,num_layers=layers,batch_first=True,
                          dropout=cfg['dropout'] if layers>1 else 0.)
        self.drop=nn.Dropout(cfg['dropout'])
        self.head=nn.Linear(hidden,cfg['num_classes'])
    def forward(self,x):
        out,_=self.lstm((x-self.mean)/self.std)
        return self.head(self.drop(out[:,-1]))
