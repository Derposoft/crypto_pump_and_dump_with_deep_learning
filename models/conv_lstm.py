import torch
import torch.nn as nn
import torch.nn.functional as F
from models.layernorm_lstm import script_lnlstm

class ConvLSTM(nn.Module):
    def __init__(self, num_feats, conv_kernel_size, embedding_size, num_layers,
        dropout=0.0, out_norm=False, cell_norm=False):
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
        self.dropout = dropout
        self.out_norm = out_norm
        self.cell_norm = cell_norm

        # encoding
        self.conv1 = nn.Conv1d(
            in_channels=num_feats,
            out_channels=embedding_size,
            kernel_size=conv_kernel_size,
        ) # out=10, kernel=5, stride=1 used in paper
        self.relu1 = nn.ReLU()
        self.pool = nn.MaxPool1d(2, 1)

        # detecting
        if cell_norm:
            self.lstm = script_lnlstm(embedding_size, embedding_size, num_layers) # no dropout for layernorm lstm; not implemented
        else:
            self.lstm = nn.LSTM(embedding_size, embedding_size, num_layers, batch_first=True, dropout=dropout)
        self.ln = nn.LayerNorm(self.embedding_size)

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
        if self.cell_norm:
            zeros = torch.zeros(y.size(1), self.embedding_size, dtype=y.dtype, device=y.device)
            hx = [(zeros, zeros) for _ in range(self.num_layers)]
            y, _ = self.lstm(y, hx)
        else:
            y, (hn, cn) = self.lstm(y) # defaulting to h_0, c_0 = 0, 0
        if self.out_norm:
            y = self.ln(y)

        # decode
        y = self.o_proj(y)
        return self.sigmoid(y).squeeze(2)

if __name__ == '__main__':
    bs, seq, feats = 128, 420, 8
    embed = 69
    x = torch.randn(bs, seq, feats)
    model = ConvLSTM(feats, 5, embed, 1)
    y = model(x)
    print(f'shape should be [{bs}, {seq}, 1]')
    print(f'shape is {y.shape}')
