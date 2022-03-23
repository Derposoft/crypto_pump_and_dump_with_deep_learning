import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvLSTM(nn.Module):
    def __init__(self, num_feats, conv_kernel_size, embedding_size, num_layers):
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
        return self.o_proj(y)

if __name__ == '__main__':
    bs, seq, feats = 128, 420, 8
    embed = 69
    x = torch.randn(bs, seq, feats)
    model = ConvLSTM(feats, 5, embed, 1)
    y = model(x)
    print(f'shape should be [{bs}, {seq}, 1]')
    print(f'shape is {y.shape}')
