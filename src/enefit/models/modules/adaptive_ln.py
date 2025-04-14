import torch.nn as nn



class AdaptiveLN(nn.Module):
    def __init__(self, module, working_size=128, cond_size=128):
        super().__init__()
        
        self.module = module
        
        self.ln = nn.LayerNorm(working_size, elementwise_affine=False)
        self.loc = nn.Linear(cond_size, working_size)
        self.scale = nn.Linear(cond_size, working_size)
        self.rescale = nn.Linear(cond_size, working_size)
        
        nn.init.constant_(self.loc.weight, 0)
        nn.init.constant_(self.loc.bias, 0)
        
        nn.init.eye_(self.scale.weight)
        nn.init.constant_(self.scale.bias, 0)
        
        nn.init.eye_(self.rescale.weight)
        nn.init.constant_(self.rescale.bias, 0)
        
    def forward(self, x, cond, *args, **kwargs):
        
        y = self.ln(x) * self.scale(cond) + self.loc(cond)
        
        y = self.module(y, *args, **kwargs)
        
        x = y * self.rescale(cond) + x

        return x