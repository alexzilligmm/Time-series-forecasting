import torch
import torch.nn as nn


class Ensemble(nn.Module):
    def __init__(self, mlp_hidden_size=256):
        super().__init__()
        
        self.mlp = nn.Sequential(
            nn.Linear(4, mlp_hidden_size), # it takes the 24-hour prediction from 3 models
            nn.SiLU(True),
            nn.Linear(mlp_hidden_size, 2) # it outputs the prediction for 24 hours
        )
        
        nn.init.kaiming_normal_(self.mlp[0].weight, mode='fan_out', nonlinearity='relu')
        nn.init.constant_(self.mlp[0].bias, 0)
        nn.init.xavier_normal_(self.mlp[2].weight)
        nn.init.constant_(self.mlp[2].bias, 0)
        
        
    def forward(self, outputs_from_models): # outputs_from_models has shape [num units, 24, 6]
        out = self.mlp(outputs_from_models) # out should have shape [num units, 24, 2]
        return out