import torch
import torch.nn as nn

from enefit.models.modules.encoding import PositionalEncoding1D
from enefit.models.modules.attention import AttentionBlock
from enefit.models.modules.mlp import MLP
from enefit.models.modules.adaptive_ln import AdaptiveLN



class Block(nn.Module):
    def __init__(self, working_size=128, cond_size=128, heads=4, dropout=None):
        super().__init__()
                
        self.attn1 = AdaptiveLN(AttentionBlock(working_size, heads, dropout), working_size, cond_size) 
        self.attn2 = AdaptiveLN(AttentionBlock(working_size, heads, dropout), working_size, cond_size) 
        self.attn3 = AdaptiveLN(AttentionBlock(working_size, heads, dropout), working_size, cond_size) 
        
        self.post = AdaptiveLN(MLP(working_size, dropout), working_size, cond_size)
        
        
    def forward(self, x, cond, county_mask=None, contract_mask=None):
        
        # Time-wise attention
        n, p, _, _ = x.shape
        x = self.attn1(x.flatten(0, 1), cond.flatten(0, 1))
        x = x.unflatten(0, (n,p))
        
        time_flat_cond = cond.permute(0, 2, 1, 3).flatten(0, 1)
        
        # County-wise attention
        x = self.attn2(x.permute(0, 2, 1, 3).flatten(0, 1), time_flat_cond, county_mask)
        
        # Contract-wise attention
        x = self.attn3(x, time_flat_cond, contract_mask)
        x = x.unflatten(0, (n, 24)).permute(0, 2, 1, 3)
        
        # Point-wise MLP
        x = self.post(x, cond)
        
        return x
        
        
class SelectiveTransformer(nn.Module):
    def __init__(self, hid_size=128, cond_size=128, heads=4, n_blocks=5, dropout=None):
        super().__init__()
        
        if dropout is None:
            dropout = 0
        
        assert 0 <= dropout < 1
        
        self.county_mapping = nn.Parameter(torch.as_tensor([[0] * i + [1] + [0] * (15 - i) for i in range(16)], dtype=torch.float32), requires_grad=False)
        self.contract_mapping = nn.Parameter(torch.as_tensor([[0] * i + [1] + [0] * (6 - i) for i in range(7)], dtype=torch.float32), requires_grad=False)
        self.month_mapping = nn.Parameter(torch.as_tensor([[0] * i + [1] + [0] * (11 - i) for i in range(12)], dtype=torch.float32), requires_grad=False)
        self.dw_mapping = nn.Parameter(torch.as_tensor([[0] * i + [1] + [0] * (6 - i) for i in range(7)], dtype=torch.float32), requires_grad=False)

        self.input = nn.Sequential(
            nn.Linear(2, hid_size),
            nn.SiLU(True),
            nn.Linear(hid_size, hid_size),
            nn.Dropout(dropout)
        )
        
        self.conditioning = nn.Sequential(
            nn.Linear(26 + 7 + 16, cond_size),
            nn.SiLU(True),
            nn.Linear(cond_size, cond_size),
            nn.Dropout(dropout)
        )
        
        self.time_enc = PositionalEncoding1D(hid_size)
    
        blocks = []
        
        for _ in range(n_blocks):
            blocks.append(Block(working_size=hid_size, cond_size=cond_size, heads=heads, dropout=dropout))
        
        self.blocks = nn.ModuleList(blocks)
        
        self.output = nn.Sequential(
            nn.Linear(hid_size, hid_size),
            nn.SiLU(True),
            nn.Linear(hid_size, 2)
        )
        
        nn.init.kaiming_normal_(self.input[0].weight, mode="fan_out", nonlinearity="relu")
        nn.init.constant_(self.input[0].bias, 0)
        nn.init.kaiming_normal_(self.input[2].weight, mode="fan_out", nonlinearity="linear")
        nn.init.constant_(self.input[2].bias, 0)
        
        nn.init.kaiming_normal_(self.conditioning[0].weight, mode="fan_out", nonlinearity="relu")
        nn.init.constant_(self.conditioning[0].bias, 0)
        nn.init.kaiming_normal_(self.conditioning[2].weight, mode="fan_out", nonlinearity="linear")
        nn.init.constant_(self.conditioning[2].bias, 0)
        
        nn.init.kaiming_normal_(self.output[0].weight, mode="fan_out", nonlinearity="relu")
        nn.init.constant_(self.output[0].bias, 0)
        nn.init.xavier_normal_(self.output[2].weight)
        nn.init.constant_(self.output[2].bias, 0)
    
    def bin_enc(self, cond):
        return torch.cat([self.county_mapping[cond[..., 0]], self.contract_mapping[cond[..., 1]], self.month_mapping[cond[..., 2]], self.dw_mapping[cond[..., 3]]], -1)
        
    def forward(self, x, cond, weather, county_mask, contract_mask):
        
        cond = self.bin_enc(cond)[:, :, None]
        cond = cond.expand(-1, -1, 24, -1)
        cond = torch.cat([cond, weather], -1)
        cond = self.conditioning(cond)
        
        n, p, _, _ = cond.shape
        
        x = self.input(x)
        x = x + self.time_enc(x.shape[-1], x.device)[None, None]
        
        for block in self.blocks:
            x = block(x, cond, county_mask, contract_mask)
        
        x = self.output(x)
        
        return x