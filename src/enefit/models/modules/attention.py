import math
import torch
import torch.nn as nn

from enefit.models.modules.encoding import RelativePosition


class AttentionBlock(nn.Module):
    def __init__(self, working_size=128, heads=4, dropout=None):
        super().__init__()
            
        if dropout is None:
            dropout = 0
            
        assert 0 <= dropout < 1
        
        self.q = nn.Linear(working_size, working_size)
        self.k = nn.Linear(working_size, working_size)
        self.v = nn.Linear(working_size, working_size)
        
        nn.init.xavier_normal_(self.q.weight)
        nn.init.constant_(self.q.bias, 0)
        
        nn.init.xavier_normal_(self.k.weight)
        nn.init.constant_(self.k.bias, 0)
        
        nn.init.xavier_normal_(self.v.weight)
        nn.init.constant_(self.v.bias, 0)
                
        self.attn = nn.MultiheadAttention(working_size, heads, batch_first=True, dropout=dropout)
        
        self.heads = heads
        
        self.out = nn.Sequential(
            nn.Linear(working_size, working_size),
            nn.Dropout(dropout)
        )
        
        nn.init.constant_(self.out[0].weight, 0)
        nn.init.constant_(self.out[0].bias, 0)
        
    def forward(self, x, attn_mask=None):              
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)
        
        if torch.is_tensor(attn_mask):
            attn_mask = attn_mask.expand(self.heads * q.shape[0], -1, -1)
        
        x = self.attn(q, k, v, need_weights=False, attn_mask=attn_mask)[0] #Attn
        
        return self.out(x)
    
class RelativeMultiHeadAttentionLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, dropout, device):
        super().__init__()
        
        assert hid_dim % n_heads == 0
        
        self.hid_dim = hid_dim
        self.n_heads = n_heads
        self.head_dim = hid_dim // n_heads
        self.max_relative_position = 168

        self.relative_position_k = RelativePosition(self.head_dim, self.max_relative_position)
        self.relative_position_v = RelativePosition(self.head_dim, self.max_relative_position)

        self.fc_q = nn.Linear(hid_dim, hid_dim)
        self.fc_k = nn.Linear(hid_dim, hid_dim)
        self.fc_v = nn.Linear(hid_dim, hid_dim)
        
        self.fc_o = nn.Linear(hid_dim, hid_dim)
        
        self.dropout = nn.Dropout(dropout)
        
        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim])).to(device)
        
    def forward(self, query, key, value, mask = None):
        #query = [batch size, query len, hid dim]
        #key = [batch size, key len, hid dim]
        #value = [batch size, value len, hid dim]
        batch_size = query.shape[0]
        len_k = key.shape[1]
        len_q = query.shape[1]
        len_v = value.shape[1]

        query = self.fc_q(query)
        key = self.fc_k(key)
        value = self.fc_v(value)

        r_q1 = query.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        r_k1 = key.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        attn1 = torch.matmul(r_q1, r_k1.permute(0, 1, 3, 2)) 

        r_q2 = query.permute(1, 0, 2).contiguous().view(len_q, batch_size*self.n_heads, self.head_dim)
        r_k2 = self.relative_position_k(len_q, len_k)
        attn2 = torch.matmul(r_q2, r_k2.transpose(1, 2)).transpose(0, 1)
        attn2 = attn2.contiguous().view(batch_size, self.n_heads, len_q, len_k)
        attn = (attn1 + attn2) / self.scale

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e10)

        attn = self.dropout(torch.softmax(attn, dim = -1))

        #attn = [batch size, n heads, query len, key len]
        r_v1 = value.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        weight1 = torch.matmul(attn, r_v1)
        r_v2 = self.relative_position_v(len_q, len_v)
        weight2 = attn.permute(2, 0, 1, 3).contiguous().view(len_q, batch_size*self.n_heads, len_k)
        weight2 = torch.matmul(weight2, r_v2)
        weight2 = weight2.transpose(0, 1).contiguous().view(batch_size, self.n_heads, len_q, self.head_dim)

        x = weight1 + weight2
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(batch_size, -1, self.hid_dim)       
        x = self.fc_o(x)       
        return x

class dotAttention(nn.Module):
    def __init__(self, hidden, time):
        super().__init__()
        #self.q_proj = nn.Parameter(torch.empty((time, hidden), requires_grad=True, device=device), requires_grad=True)
        #self.k_proj = nn.Parameter(torch.empty((time, hidden), requires_grad=True, device=device), requires_grad=True)
        #self.v_proj = nn.Parameter(torch.empty((time, hidden), requires_grad=True, device=device), requires_grad=True)
        #xavier_normal_(self.q_proj)
        #xavier_normal_(self.k_proj)
        #xavier_normal_(self.v_proj)
        
        self.scale = 1/math.sqrt(hidden)

    def forward(self, q, k, v):

        #q = torch.einsum('bhf, hf -> bhf', q, self.q_proj)
        #k = torch.einsum('bhf, hf -> bhf', k, self.k_proj)
        #v = torch.einsum('bhf, hf -> bhf', v, self.v_proj)

        q = q.transpose(1,2)
        att = torch.einsum('bji, bij -> bi', q, k) * self.scale
        att = torch.clamp(att, min = -60, max = 60)
        soft = torch.softmax(att, dim = 1)
        return torch.einsum('bi, bkw ->bw', soft, v)