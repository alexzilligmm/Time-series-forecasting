import torch
import torch.nn as nn
import numpy as np

from enefit import DEVICE

# Positional encoding repurposed from tatp22/multidim-positional-encoding

def get_emb(sin_inp):
    """
    Gets a base embedding for one dimension with sin and cos intertwined
    """
    emb = torch.stack((sin_inp.sin(), sin_inp.cos()), dim=-1)
    return torch.flatten(emb, -2, -1)


class PositionalEncoding1D(nn.Module):
    def __init__(self, channels):
        """
        :param channels: The last dimension of the tensor you want to apply pos emb to.
        """
        super(PositionalEncoding1D, self).__init__()
        self.org_channels = channels
        channels = int(np.ceil(channels / 2) * 2)
        self.channels = channels
        inv_freq = 1.0 / (10000 ** (torch.arange(0, channels, 2).float() / channels))
        self.register_buffer("inv_freq", inv_freq)
        self.register_buffer("cached_penc", None)

    def forward(self, size, device="cpu"):
        """
        :param tensor: A size s
        :return: Positional Encoding Matrix of size (24, s)
        """

        if self.cached_penc is not None and self.cached_penc.shape[-1] == size:
            return self.cached_penc

        self.cached_penc = None
        pos_x = torch.arange(24, device=device).type(self.inv_freq.type())
        sin_inp_x = torch.einsum("i,j->ij", pos_x, self.inv_freq)
        emb_x = get_emb(sin_inp_x)
        emb = torch.zeros((24, self.channels), device=device, dtype=torch.float32)
        emb[:, : self.channels] = emb_x

        self.cached_penc = emb[..., :size]
        return self.cached_penc
    
class PositionalEncoding(nn.Module):
    def __init__(self, embedding_size=32, max_sequence_length=336):
        super(PositionalEncoding, self).__init__()
        self.encoding = torch.zeros(max_sequence_length, embedding_size)
        position = torch.arange(0, max_sequence_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embedding_size, 2).float() * -(torch.log(torch.tensor(10000.0)) / embedding_size))
        self.encoding[:, 0::2] = torch.sin(position * div_term)
        self.encoding[:, 1::2] = torch.cos(position * div_term)
        self.encoding = self.encoding.unsqueeze(0)

    def forward(self, x):
        return x + self.encoding[:, :x.size(1)].detach().to(DEVICE) # idk why I have to say to(device) here, otherwise it's on the cpu
    
    
class RelativePosition(nn.Module):

    def __init__(self, num_units, max_relative_position):
        super().__init__()
        self.num_units = num_units
        self.max_relative_position = max_relative_position
        self.embeddings_table = nn.Parameter(torch.Tensor(max_relative_position * 2 + 1, num_units))
        nn.init.xavier_uniform_(self.embeddings_table)

    def forward(self, length_q, length_k):
        range_vec_q = torch.arange(length_q)
        range_vec_k = torch.arange(length_k)
        distance_mat = range_vec_k[None, :] - range_vec_q[:, None]
        distance_mat_clipped = torch.clamp(distance_mat, -self.max_relative_position, self.max_relative_position)
        final_mat = distance_mat_clipped + self.max_relative_position
        final_mat = torch.LongTensor(final_mat)
        embeddings = self.embeddings_table[final_mat]

        return embeddings