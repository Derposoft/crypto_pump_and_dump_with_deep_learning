import torch
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

#########################################
##### Feature expansion Transformer #####
#########################################
class TransformerTimeSeriesExpandedFeatures(nn.Module):
    def __init__(self, feature_size, output_size, nhead=2, num_layers=2, dropout=0.1, input_size=8, non_expansion_size=5):
        super().__init__()
        self.input_size = input_size
        self.feature_expander = nn.Linear(input_size, feature_size-non_expansion_size)
        self.layers = nn.TransformerEncoderLayer(d_model=feature_size, nhead=nhead, batch_first=True, dropout=dropout)
        self.transformer = nn.TransformerEncoder(self.layers, num_layers=num_layers)
        self.decoder = nn.Linear(feature_size, output_size)
    
    def forward(self, x):
        # X shape: [batch_size, seq_len, feature_size]
        x_feat = x[:,:,:self.input_size]
        x_feat = self.feature_expander(x_feat)

        x = torch.cat((x_feat, x[:,:,self.input_size:]),dim=2)

        x = self.transformer(x)
        # X shape: [batch_size, seq_len, feature_size]

        x = self.decoder(x)
        # X shape: [batch_size, seq_len, output_size]
        
        return x