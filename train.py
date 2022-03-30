import torch
from torch.utils.data import DataLoader
from models.conv_lstm import *
import pandas as pd
import numpy as np


def train(model, iterator, opt, criterion, device):
    epoch_loss = 0
    n_seq = 2
    for batch in iterator:
        opt.zero_grad()
        x = batch[:, :-1].chunk(n_seq)
        pred = model(torch.stack(list(x), dim=0).double().to(device))
        loss = criterion(torch.flatten(pred[:, :, -1]).to(device), batch[:, -1].to(device))
        loss.backward()
        opt.step()

        epoch_loss += loss.item()
        print("Batch Loss" + str(loss.item()))
    return epoch_loss / len(iterator)


def main():
    data = pd.read_csv('data/features_5S.csv.gz', compression='gzip')
    data = data.drop(columns=['symbol', 'date']).to_numpy()
    data = torch.from_numpy(data)
    data = data[:800000, :]

    EMBEDDING_SIZE = 64
    N_LAYERS = 1
    LEARNING_RATE = .01
    N_EPOCHS = 10
    KERNEL_SIZE = 1
    BATCH_SIZE = 100000

    train_loader = DataLoader(data, batch_size = BATCH_SIZE)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = ConvLSTM(13, KERNEL_SIZE, EMBEDDING_SIZE, N_LAYERS)
    model.double()
    model.to(device)
    opt = torch.optim.Adam(model.parameters())
    criterion = torch.nn.BCEWithLogitsLoss().to(device)

    for epoch in range(N_EPOCHS):
        print('Epoch: ' + str(epoch))
        train_loss = train(model, train_loader, opt, criterion, device)
        print('ELoss: ' + str(train_loss))
