import torch.nn as nn

class TransformerTimeSeries(nn.Module):
    def __init__(self, feature_size, output_size, nhead=2, num_layers=2, dropout=0.1):
        super().__init__()
        self.layers = nn.TransformerEncoderLayer(d_model=feature_size, nhead=nhead, batch_first=True, dropout=dropout)
        self.transformer = nn.TransformerEncoder(self.layers, num_layers=num_layers)
        self.decoder = nn.Linear(feature_size, output_size)
    
    def forward(self, x):
        # X shape: [batch_size, seq_len, feature_size]

        x = self.transformer(x)
        # X shape: [batch_size, seq_len, feature_size]

        x = self.decoder(x)
        # X shape: [batch_size, seq_len, output_size]
        
        return x