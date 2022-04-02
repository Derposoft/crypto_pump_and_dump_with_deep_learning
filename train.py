from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
import pandas as pd
import math

import torch
from torch.utils.data import DataLoader
from models.conv_lstm import ConvLSTM
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# conv model hyperparameters
EMBEDDING_SIZE = 128
N_LAYERS = 20
N_EPOCHS = 50
KERNEL_SIZE = 5

# train hyperparameters
LEARNING_RATE = 1e-3
N_SEQ = 100
BATCH_SIZE = 128
P_R_THRESHOLD = 0.3031 # precision-recall threshold
TRAIN_RATIO = 0.5 # [0.0, 1.0] proportion of data used for train


def train(model, dataloader, opt, criterion, device):
    '''
    trains given model with given dataloader, optimizer, criterion, and on device.
    :returns: loss
    '''
    epoch_loss = 0
    for batch in dataloader:
        # training step
        opt.zero_grad()
        x = batch[:,:,:-1].to(device)
        y = batch[:,:,-1].to(device)
        preds = model(x).reshape(y.shape) # goes from (batch_size, seq_len, 1) -> (batch_size, seq_len)
        loss = criterion(preds, y)
        loss.backward()
        opt.step()
        epoch_loss += loss.item()
    return epoch_loss/len(dataloader)

def validate(model, dataloader, device):
    accuracy = 0
    f1 = 0
    recall = 0
    precision = 0
    n_batches = len(dataloader)
    for batch in dataloader:
        with torch.no_grad():
            x = batch[:, :, :-1].to(device)
            y = batch[:, :, -1].to(device)
            preds = (model(x).reshape(y.shape) > P_R_THRESHOLD).float()
            y, preds = y.cpu().flatten(), preds.cpu().flatten()
            accuracy += accuracy_score(y, preds)
            f1 += f1_score(y, preds, zero_division=1)
            recall += recall_score(y, preds, zero_division=1)
            precision += precision_score(y, preds, zero_division=1)
    return accuracy/n_batches, f1/n_batches, recall/n_batches, precision/n_batches

if __name__ == '__main__':
    data = pd.read_csv('data/features_5S.csv.gz', compression='gzip').drop(columns=['symbol', 'date'])
    n_feats = data.shape[1]-1 # since the last column is the target value
    train_data, test_data = train_test_split(data, train_size=TRAIN_RATIO, shuffle=False)
    train_data = torch.FloatTensor(train_data.to_numpy()).chunk(math.ceil(len(train_data)/N_SEQ))
    test_data = torch.FloatTensor(test_data.to_numpy()).chunk(math.ceil(len(test_data)/N_SEQ))
    train_loader = DataLoader(train_data, batch_size = BATCH_SIZE, drop_last=True)
    test_loader = DataLoader(test_data, batch_size = BATCH_SIZE, drop_last=True)
    
    conv_model = ConvLSTM(n_feats, KERNEL_SIZE, EMBEDDING_SIZE, N_LAYERS).to(device)
    conv_opt = torch.optim.Adam(conv_model.parameters(), lr=LEARNING_RATE)
    #criterion = torch.nn.BCEWithLogitsLoss().to(device)
    criterion = torch.nn.MSELoss().to(device)
    for epoch in range(N_EPOCHS):
        loss = train(conv_model, train_loader, conv_opt, criterion, device)
        acc, f1, recall, precision = validate(conv_model, test_loader, device)
        print(f'Epoch: {epoch}')
        print(f'Train -- Loss: {loss:0.5f}')
        print(f'Val   -- Acc: {acc:0.5f} -- Precision: {precision:0.5f} -- Recall: {recall:0.5f} -- F1: {f1:0.5f}')


