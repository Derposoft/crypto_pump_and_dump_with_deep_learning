import torch.nn as nn
import torch.nn.functional as F

class ConvLSTM(nn.Module):
    def __init__(self, num_features, conv_kernel_size, embedding_size, num_layers):
        # encoding
        self.conv1 = nn.Conv1d(
            in_channels=num_features,
            out_channels=embedding_size,
            kernel_size=conv_kernel_size,
            padding=(conv_kernel_size-1, 0),
            padding_mode='replicate'
        ) # out=10, kernel=5, stride=1 used in paper
        self.relu1 = nn.ReLU()
        self.pool = nn.MaxPool1d(2, 1)

        # lstm
        self.lstm = nn.LSTM(embedding_size, embedding_size, num_layers, batch_first=True)

        # decoding
        #self.conv2 = nn.Conv1d(in_channels=embedding_size, out_channels=1, kernel_size=conv_kernel_size)
        self.o_proj = nn.Linear(embedding_size, 1)
        

    def forward(self, y):
        # encode
        y = self.conv1(y)
        y = self.relu1(y)
        y = self.pool(y)

        # detection
        y, (hn, cn) = self.lstm(y) # defaulting to h_0, c_0 = 0, 0

        # decode
        #y = self.conv2(y)
        y = self.o_proj(y)
        return y
