import torch
from torch.utils.data import DataLoader
from models.conv_lstm import *
import pandas as pd


def train(model, iterator, opt, criterion):
    epoch_loss = 0
    n_seq = 2
    for batch in iterator:
        opt.zero_grad()
        x = batch[:, :-1].chunk(n_seq)
        pred = model(torch.stack(list(x), dim=0))
        loss = criterion(pred[:, :, -1], batch[:, -1])
        loss.backward()
        opt.step()

        epoch_loss += loss.item()
        print("Batch Loss")
    return epoch_loss / len(iterator)


def main():
    data = pd.read_csv('data/features_5S.csv.gz', compression='gzip')
    data = data.drop(columns=['symbol', 'date']).to_numpy()
    

    EMBEDDING_SIZE = 64
    N_LAYERS = 1
    LEARNING_RATE = .01
    N_EPOCHS = 10
    KERNEL_SIZE = 1
    BATCH_SIZE = 10

    train_loader = DataLoader(data, batch_size = BATCH_SIZE)

    model = ConvLSTM(13, KERNEL_SIZE, EMBEDDING_SIZE, N_LAYERS)
    opt = torch.optim.Adam(model.parameters())
    criterion = torch.nn.BCEWithLogitsLoss()

    for epoch in range(N_EPOCHS):
        print('Epoch: ' + str(epoch))
        train_loss = train(model, train_loader, opt, criterion)
        print('ELoss: ' + str(train_loss))
