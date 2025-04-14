import torch
import torch.nn as nn

from enefit import DEVICE

from enefit.models.modules.attention import RelativeMultiHeadAttentionLayer
from enefit.models.modules.embedder import Embedder
from enefit.models.modules.encoding import PositionalEncoding


class EncoderBlock(nn.Module):
    def __init__(self, embd_size, d_model=None, num_heads=2, dropout_p=0.15):
        if d_model == None: d_model = embd_size*num_heads
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        nn.init.constant_(self.norm1.weight, 1)
        nn.init.constant_(self.norm1.bias, 0)
        
        self.q = nn.Linear(d_model, embd_size * num_heads)
        self.k = nn.Linear(d_model, embd_size * num_heads)
        self.v = nn.Linear(d_model, embd_size * num_heads)
        
        nn.init.xavier_normal_(self.q.weight)
        nn.init.constant_(self.v.bias, 0)
        nn.init.xavier_normal_(self.k.weight)
        nn.init.constant_(self.k.bias, 0)
        nn.init.xavier_normal_(self.v.weight)
        nn.init.constant_(self.v.bias, 0)
        
        self.multiheadattention = nn.MultiheadAttention(embd_size*num_heads, num_heads, batch_first=True)
        
        self.norm2 = nn.LayerNorm(embd_size * num_heads)
        nn.init.constant_(self.norm2.weight, 1)
        nn.init.constant_(self.norm2.bias, 0)
        
        self.mlp = nn.Sequential(
            nn.Linear(embd_size * num_heads, embd_size * num_heads),
            nn.SiLU(),
            nn.Linear(embd_size * num_heads, embd_size * num_heads)
        )
        
        nn.init.kaiming_normal_(self.mlp[0].weight)
        nn.init.constant_(self.mlp[0].bias, 0)
        nn.init.xavier_normal_(self.mlp[2].weight)
        nn.init.constant_(self.mlp[2].bias, 0)
    
    def forward(self, x):
        # Sub-layer 1: Multihead Attention
        residual = x
        #print("x shape: ", x.shape)
        x = self.norm1(x)
        q = self.q(x)
        #print("q shape: ", q.shape)
        k = self.k(x)
        #print("k shape: ", q.shape)
        v = self.v(x)
        #print("v shape: ", q.shape)
        new_v, _ = self.multiheadattention(q, k, v)
        #print("multihead_output shape: ", new_v.shape)
        x = new_v + residual
        
        # Sub-layer 2: Feedforward Neural Network
        residual = x
        x = self.norm2(x)
        mlp_output = self.mlp(x)
        x = residual + mlp_output
        return x
    
    
    
class RelativeAttentionEncoderBlock(nn.Module):
    def __init__(self, embd_size, d_model=None, num_heads=2, dropout_p=0.15):
        if d_model == None: d_model = embd_size*num_heads
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        nn.init.constant_(self.norm1.weight, 1)
        nn.init.constant_(self.norm1.bias, 0)
        
        self.q = nn.Linear(d_model, embd_size * num_heads)
        self.k = nn.Linear(d_model, embd_size * num_heads)
        self.v = nn.Linear(d_model, embd_size * num_heads)
        
        nn.init.xavier_normal_(self.q.weight)
        nn.init.constant_(self.v.bias, 0)
        nn.init.xavier_normal_(self.k.weight)
        nn.init.constant_(self.k.bias, 0)
        nn.init.xavier_normal_(self.v.weight)
        nn.init.constant_(self.v.bias, 0)
        
        
        self.multiheadattention = RelativeMultiHeadAttentionLayer(hid_dim=d_model, n_heads=num_heads, dropout=dropout_p, device=DEVICE).to(DEVICE)
        
        self.norm2 = nn.LayerNorm(embd_size * num_heads)
        nn.init.constant_(self.norm2.weight, 1)
        nn.init.constant_(self.norm2.bias, 0)
        
        self.mlp = nn.Sequential(
            nn.Linear(embd_size * num_heads, embd_size * num_heads),
            nn.SiLU(),
            nn.Linear(embd_size * num_heads, embd_size * num_heads)
        )
        
        nn.init.kaiming_normal_(self.mlp[0].weight)
        nn.init.constant_(self.mlp[0].bias, 0)
        nn.init.xavier_normal_(self.mlp[2].weight)
        nn.init.constant_(self.mlp[2].bias, 0)
    
    def forward(self, x):
        # Sub-layer 1: Multihead Attention
        residual = x
        #print("x shape: ", x.shape)
        x = self.norm1(x)
        q = self.q(x)
        #print("q shape: ", q.shape)
        k = self.k(x)
        #print("k shape: ", q.shape)
        v = self.v(x)
        #print("v shape: ", q.shape)
        new_v = self.multiheadattention(q, k, v)
        #print("multihead_output shape: ", new_v.shape)
        x = new_v + residual
        
        # Sub-layer 2: Feedforward Neural Network
        residual = x
        x = self.norm2(x)
        mlp_output = self.mlp(x)
        x = residual + mlp_output
        return x
    
    
class Encoder(nn.Module):
    def __init__(self, embd_size, d_model=None, num_heads=2, dropout_p=0.15, num_enc_blocks=2, modality="vanilla"):
        super().__init__()
        if modality == "relative":
            self.enc_layers = nn.Sequential(*[RelativeAttentionEncoderBlock(embd_size, d_model, num_heads, dropout_p) for _ in range(num_enc_blocks)])
        else:
            self.enc_layers = nn.Sequential(*[EncoderBlock(embd_size, d_model, num_heads, dropout_p) for _ in range(num_enc_blocks)])
            
    def forward(self, x):
        x = self.enc_layers(x)
        return x
    
    
class AutoregressiveLSTMDecoder(nn.Module):
    def __init__(self, emb_size, hidden_size, output_size=2, num_layers=1, dropout_p=0.15):
        super(AutoregressiveLSTMDecoder, self).__init__()

        self.lstm = nn.LSTM(input_size=emb_size, 
                            hidden_size=emb_size, 
                            num_layers=num_layers,
                            dropout = dropout_p,
                            batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        
        nn.init.kaiming_normal_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0)

    def forward(self, x, sequence_length):
        h0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(x.device)
        c0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(x.device)

        output_sequence = []
        for t in range(sequence_length):
            # Forward pass through the LSTM
            lstm_out, (hidden, cell) = self.lstm(x[:, t%x.shape[1], :].unsqueeze(1), (h0, c0))
            # Apply the fully connected layer to get the output at each time step
            output_t = self.fc(lstm_out.squeeze(1))
            output_sequence.append(output_t)

            # Update the hidden state and cell state for the next time step
            h0 = hidden
            c0 = cell

        # Stack the output_sequence along the time dimension
        output_sequence = torch.stack(output_sequence, dim=1)

        return output_sequence #,(hidden, cell)
    
    
class MyTSFTransformer(nn.Module):
    def __init__(self, embd_size=16, d_model=None, num_heads=2, dropout_p=0.15, 
                 num_enc_blocks=2, num_dec_layers=2, output_size=48, modality="vanilla"):
        super().__init__()
        if d_model == None: d_model = embd_size * num_heads
        self.embedder = Embedder(num_counties=16, num_product_types=4, 
                    embedding_dim_county=4, embedding_dim_business=2, embedding_dim_product=2, 
                    embedding_dim_month=6, embedding_dim_weekday=3, embedding_dim_hour=5)
        self.emb_to_hidden = nn.Linear(32, d_model)
        self.positional_encoder = PositionalEncoding(d_model)
        self.encoder = Encoder(embd_size=embd_size, d_model=d_model, num_heads=num_heads, 
                               dropout_p=dropout_p, num_enc_blocks=num_enc_blocks, modality=modality)
        self.decoder = AutoregressiveLSTMDecoder(emb_size=d_model, hidden_size=d_model, 
                                                 output_size=2, num_layers=num_dec_layers)
        self.output_size=output_size
        
    def forward(self, x):
        #print("x shape: ", x.shape)
        y = self.embedder(x)
        #print("embedded x shape: ", x.shape)
        y = self.emb_to_hidden(y)
        #print("embedded x to d_model shape: ", x.shape)
        y = self.positional_encoder(y)
        #print("positional encoded x to d_model shape: ", x.shape)
        y = self.encoder(y)
        #print("encoder encoded x shape: ", x.shape)
        out = self.decoder(y, self.output_size)
        #print("decoder decoded x shape: ", out.shape)
        
        return out