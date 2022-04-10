from typing import List, Tuple, Optional, overload
from torch import Tensor
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import LSTMCell

#from models.custom_lstms import LayerNormLSTMCell, StackedLSTMWithDropout, LSTMLayer

class ConvLSTM(nn.Module):
    def __init__(self, num_feats, conv_kernel_size, embedding_size, num_layers, dropout=0.0):
        '''
        inspired by https://ieeexplore.ieee.org/abstract/document/9382507
        intput size should be (batch_size, seq_len, num_feats)
        output size should be (batch_size, seq_len, 1)
        '''
        super(ConvLSTM, self).__init__()
        self.num_feats = num_feats
        self.conv_kernel_size = conv_kernel_size
        self.embedding_size = embedding_size
        self.num_layers = num_layers

        # encoding
        self.conv1 = nn.Conv1d(
            in_channels=num_feats,
            out_channels=embedding_size,
            kernel_size=conv_kernel_size,
        ) # out=10, kernel=5, stride=1 used in paper
        self.relu1 = nn.ReLU()
        self.pool = nn.MaxPool1d(2, 1)

        # detecting
        self.lstm = nn.LSTM(embedding_size, embedding_size, num_layers, batch_first=True)


        # decoding
        self.o_proj = nn.Linear(embedding_size, 1)
        self.sigmoid = nn.Sigmoid()
        

    def forward(self, x):
        '''
        x: (batch_size, seq_len, num_feats)
        '''
        # encode
        y = torch.permute(x, [0, 2, 1]) # conv/pool input=(batch_size, in_channels, seq_len)
        y = F.pad(y, (self.conv_kernel_size,0), 'replicate') # ensure input/output seq lens are same since we're doing n:n
        y = self.conv1(y)
        y = self.relu1(y)
        y = self.pool(y)

        # detect
        y = torch.permute(y, [0, 2, 1]) # lstm input=(batch_size, seq_len, num_feats)
        y, (hn, cn) = self.lstm(y) # defaulting to h_0, c_0 = 0, 0

        # decode
        y = self.o_proj(y)
        return self.sigmoid(y)

class LSTMCell(nn.LSTMCell):
    def __init__(self, input_size, hidden_size, bias=True):
        super(nn.LSTMCell, self).__init__(input_size, hidden_size, bias)
        self.layernorm_i = nn.LayerNorm(4 * hidden_size)
        self.layernorm_h = nn.LayerNorm(4 * hidden_size)
        self.layernorm_c = nn.LayerNorm(hidden_size)
    
    @overload
    def forward(self, input: Tensor, hx: Optional[Tuple[Tensor, Tensor]] = None) -> Tuple[Tensor, Tensor]:
        # error checking/initial states (from torch.nn.modules.rnn.py)
        assert input.dim() in (1, 2), \
            f"LSTMCell: Expected input to be 1-D or 2-D but received {input.dim()}-D tensor"
        is_batched = input.dim() == 2
        if not is_batched:
            input = input.unsqueeze(0)
        if hx is None:
            zeros = torch.zeros(input.size(0), self.hidden_size, dtype=input.dtype, device=input.device)
            hx = (zeros, zeros)
        else:
            hx = (hx[0].unsqueeze(0), hx[1].unsqueeze(0)) if not is_batched else hx
        
        # lstm cell logic
        hx, cx = hx
        igates = self.layernorm_i(F.linear(input, self.weight_ih.t()))
        hgates = self.layernorm_h(F.linear(hx, self.weight_hh.t()))
        gates = igates + hgates
        ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

        ingate = F.sigmoid(ingate)
        forgetgate = F.sigmoid(forgetgate)
        cellgate = F.tanh(cellgate)
        outgate = F.sigmoid(outgate)

        cy = self.layernorm_c((forgetgate * cx) + (ingate * cellgate))
        hy = outgate * F.tanh(cy)

        return hy, (hy, cy)



if __name__ == '__main__':
    bs, seq, feats = 128, 420, 8
    embed = 69
    x = torch.randn(bs, seq, feats)
    model = ConvLSTM(feats, 5, embed, 1)
    y = model(x)
    print(f'shape should be [{bs}, {seq}, 1]')
    print(f'shape is {y.shape}')
