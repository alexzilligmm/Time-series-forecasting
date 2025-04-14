import torch
import torch.nn as nn

from enefit import DEVICE
from enefit.models.modules.attention import dotAttention



class lstmEncoder(nn.Module):
    def __init__(self, n_feature, hidden_size):
        super().__init__()
        self.lstm = nn.LSTM(n_feature, hidden_size, num_layers=1, bidirectional=True, batch_first=True, device=DEVICE)
        self.hidden_size = hidden_size
        
        self.init_hx = nn.Parameter(torch.empty((2, 1, hidden_size)), requires_grad=True)
        self.init_cx = nn.Parameter(torch.empty((2, 1, hidden_size)), requires_grad=True)
        
        nn.init.xavier_normal_(self.init_hx)
        nn.init.xavier_normal_(self.init_cx)
        
        for name, param in self.lstm.named_parameters():
            if "weight" in name:
                nn.init.xavier_normal_(param)
            if "bias" in name:
                nn.init.constant_(param, 0)
    
    def forward(self, x):
        B, T, F = x.shape
        h_n = self.init_hx.expand(-1, B, -1).contiguous()
        c_n = self.init_cx.expand(-1, B, -1).contiguous()
        output, (h_n, c_n) = self.lstm(x, (h_n, c_n))

        return output, (h_n, c_n)

class lstmDecoder(nn.Module):
    def __init__(self, enc_out, hidden_size, time):
        super().__init__()

        self.hidden_size = hidden_size
        self.time = time
        self.attn = dotAttention(2*self.hidden_size, time)
        self.lstm = nn.LSTM(enc_out, hidden_size, num_layers=2, batch_first=True, device=DEVICE)
        
        
        for name, param in self.lstm.named_parameters():
            if "weight" in name:
                nn.init.xavier_normal_(param)
            if "bias" in name:
                nn.init.constant_(param, 0)
        
        #self.linear = nn.Linear(hidden_size, enc_out)
        
        #nn.init.xavier_normal_(self.linear.weight)
        #nn.init.constant_(self.linear.bias, 0)
    
    def forward(self, dec_input, enc_out, dec_hids):
        B, T, F = enc_out.shape
        # unwrap the hids 
        h_n, _ = dec_hids

        # compute the s_n
        s_n =  h_n.view(B, 1, F).repeat(1, self.time, 1)
        # compute the context based on the encoder output
        c_n = self.attn(s_n, enc_out, enc_out)
        # reshape to fit the decoder
        c_n = c_n.view(2, B, self.hidden_size)

        # decode it 
        output, (h_n, c_n) = self.lstm(dec_input.unsqueeze(1), (h_n, c_n))
        # post proc
        output = output.squeeze(1)
        #output = self.linear(output.squeeze(1))   

        return output, (h_n, c_n)
    
    
class lstmEncAttDec(nn.Module):
    def __init__(self, n_feature, hidden_size, time):
        super().__init__()
        
        self.preproc = nn.Sequential(
            nn.Linear(n_feature, hidden_size//2),
            nn.LeakyReLU(.1, True),
            nn.Linear(hidden_size//2, hidden_size)
        )

        nn.init.kaiming_normal_(self.preproc[0].weight, a=.1, mode="fan_out", nonlinearity="leaky_relu")
        nn.init.constant_(self.preproc[0].bias, 0)
        nn.init.xavier_normal_(self.preproc[2].weight)
        nn.init.constant_(self.preproc[2].bias, 0)
        

        self.encoder = lstmEncoder(hidden_size, hidden_size)
        self.decoder = lstmDecoder(hidden_size, hidden_size, time)

        self.postproc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size//2),
            nn.LeakyReLU(.1, True),
            nn.Linear(hidden_size//2, 2)
        )

        nn.init.kaiming_normal_(self.postproc[0].weight, a=.1, mode="fan_out", nonlinearity="leaky_relu")
        nn.init.constant_(self.postproc[0].bias, 0)
        nn.init.xavier_normal_(self.postproc[2].weight)
        nn.init.constant_(self.postproc[2].bias, 0)

    def forward(self, x, n_pred):
        
        # embeddings
        embs = self.preproc(x)
        B, T, F = embs.shape
        # encode 
        enc_out, enc_hids = self.encoder(embs)
        # reshape the input to fit the decoder 
        dec_input = enc_out.view(B, T, F, 2).sum(-1)
        # prepare the output 
        outputs = []
        # initialize the decoder hids
        dec_hids = enc_hids

        for t in range(n_pred):
            # use le last n_pred-t encore output to make the predcitions
            output, dec_hids = self.decoder(dec_input[:, -(n_pred -t), :], enc_out, dec_hids)
            outputs.append(output)
            
        outputs = torch.stack(outputs, 1)
        
        return self.postproc(outputs)