import torch
import torch.nn as nn
import torch.nn.functional as F

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
 
        return x.squeeze(2)

    def loss_fn(self, y_pred, y_true):
        gaussian_filter = torch.nn.Conv1d(1, 1, kernel_size=5, padding=2, bias=False)
        gaussian_filter.weight.data = torch.tensor([[[0.05, 0.25, 0.4, 0.25, 0.05]]])
        gaussian_filter.weight.requires_grad = False
        y_gaussian_filter = gaussian_filter(y_true.unsqueeze(1)).squeeze(1)

        return F.mse_loss(y_pred, y_true) + F.mse_loss(y_pred, y_gaussian_filter)

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
        
        return x.squeeze(2)