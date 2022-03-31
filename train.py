import torch
from torch.utils.data import DataLoader
from models.conv_lstm import ConvLSTM
import pandas as pd
import math

EMBEDDING_SIZE = 64
N_LAYERS = 1
LEARNING_RATE = 1e-3
N_EPOCHS = 10
KERNEL_SIZE = 1
BATCH_SIZE = 100000
N_SEQ = 2
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train(model, iterator, opt, criterion, device):
    epoch_loss = 0
    for batch in iterator:
        opt.zero_grad()
        x = batch[:,:,:-1].to(device)
        y = batch[:,:,-1].to(device)
        preds = model(x).reshape(y.shape) # goes from (batch_size, seq_len, 1) -> (batch_size, seq_len)
        loss = criterion(preds, y)
        loss.backward()
        opt.step()
        epoch_loss += loss.item()
        #print(f'Batch Loss: {loss.item()}') # commenting this out because it goes too fast and gets spammy
    return epoch_loss / len(iterator)

if __name__ == '__main__':
    data = pd.read_csv('data/features_5S.csv.gz', compression='gzip').drop(columns=['symbol', 'date'])
    data = torch.FloatTensor(data.to_numpy()).chunk(math.ceil(len(data)/N_SEQ))
    n_feats = data[0].shape[1]-1 # since the last column is the target value
    train_loader = DataLoader(data, batch_size = BATCH_SIZE, drop_last=True)
    
    conv_model = ConvLSTM(n_feats, KERNEL_SIZE, EMBEDDING_SIZE, N_LAYERS).to(device)
    conv_opt = torch.optim.Adam(conv_model.parameters(), lr=LEARNING_RATE)
    criterion = torch.nn.BCEWithLogitsLoss().to(device)
    for epoch in range(N_EPOCHS):
        train_loss = train(conv_model, train_loader, conv_opt, criterion, device)
        print(f'Epoch: {epoch} Loss: {train_loss}')
