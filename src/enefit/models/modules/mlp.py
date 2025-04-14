import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, working_size=128, dropout=None):
        super().__init__()
        
        if dropout is None:
            dropout = 0
        
        assert 0 <= dropout < 1
        
        self.out = nn.Sequential(
            nn.Linear(working_size, working_size),
            nn.SiLU(True),
            nn.Linear(working_size, working_size),
            nn.Dropout(dropout)
        )
        
        nn.init.kaiming_normal_(self.out[0].weight, mode="fan_out", nonlinearity="relu")
        nn.init.constant_(self.out[0].bias, 0)
        nn.init.constant_(self.out[2].weight, 0)
        nn.init.constant_(self.out[2].bias, 0)
        
    def forward(self, x):
        return self.out(x)